//! Searching ArUco detector and preprocessing parameters for a dataset.
//!
//! Fiducial detection is unusually sensitive to capture conditions: the
//! adaptive-threshold window has to straddle the marker's black border, and
//! the threshold offset has to survive whatever exposure the scene was shot
//! at. One default cannot serve a brightly-lit calibration board and a dim
//! machine-vision frame equally well, and the failure is silent - markers
//! simply aren't found.
//!
//! `sfmtory feature --find-params` sweeps the parameters that actually matter
//! against a sample of the real dataset and keeps the best-scoring
//! combination, which is then saved into the project so ordinary runs pick it
//! up automatically.

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;
use sfm_features::aruco::ArucoParams;

use crate::dataset::DiscoveredImage;

/// Images sampled from the dataset for the sweep. Detection is the expensive
/// part and parameter quality is a property of the capture setup rather than
/// of any one frame, so a spread across the dataset is as informative as all
/// of it and far cheaper.
const MAX_SAMPLE_IMAGES: usize = 12;

/// Score for one candidate: total markers found, but counting only images that
/// found at least one, and penalising nothing else.
///
/// Maximising raw marker count alone is the right objective here - a *false*
/// marker detection is nearly impossible to produce by accident, because a
/// candidate quad only becomes a marker if its sampled bit pattern matches a
/// dictionary entry within the Hamming bound. So more detections really does
/// mean better parameters, and there is no precision/recall trade-off to
/// balance. Coverage is tracked as a tie-break so a setting that finds many
/// markers in one image loses to one that finds them across the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Score {
    images_with_markers: usize,
    total_markers: usize,
}

impl Score {
    fn is_better_than(&self, other: &Score) -> bool {
        (self.images_with_markers, self.total_markers)
            > (other.images_with_markers, other.total_markers)
    }
}

fn score_params(sample: &[&DiscoveredImage], params: &ArucoParams) -> Score {
    let per_image: Vec<usize> = sample
        .par_iter()
        .map(|d| match image::open(&d.path) {
            Ok(img) => sfm_features::aruco::detect(&img, params).len() / 4,
            Err(_) => 0,
        })
        .collect();
    Score {
        images_with_markers: per_image.iter().filter(|&&n| n > 0).count(),
        total_markers: per_image.iter().sum(),
    }
}

/// Sweeps the parameters that materially change fiducial detection and returns
/// the best-scoring combination.
pub fn find_params(images: &[DiscoveredImage]) -> Result<ArucoParams> {
    if images.is_empty() {
        anyhow::bail!("no images to tune ArUco parameters on");
    }
    // Evenly spread rather than the first N, so a dataset whose first capture
    // is unrepresentative doesn't drive the whole choice.
    let stride = (images.len() / MAX_SAMPLE_IMAGES).max(1);
    let sample: Vec<&DiscoveredImage> = images
        .iter()
        .step_by(stride)
        .take(MAX_SAMPLE_IMAGES)
        .collect();
    println!(
        "Tuning ArUco parameters on {} of {} images...",
        sample.len(),
        images.len()
    );

    let base = ArucoParams::default();
    // Adaptive-threshold window and offset are the two that decide whether a
    // marker's border is found at all; contrast and gamma stand in for the
    // exposure differences that push a capture out of the range those two
    // cover.
    let radii = [3, 5, 7, 11, 15, 21];
    let offsets = [4.0f32, 8.0, 12.0, 18.0, 25.0];
    let contrasts = [1.0f32, 1.4, 1.8];
    let gammas = [1.0f32, 0.7, 1.4];

    let mut best = base;
    let mut best_score = score_params(&sample, &base);
    println!(
        "  baseline: {} markers across {}/{} images",
        best_score.total_markers,
        best_score.images_with_markers,
        sample.len()
    );

    let mut evaluated = 1usize;
    for &r in &radii {
        for &c in &offsets {
            for &contrast in &contrasts {
                for &gamma in &gammas {
                    let cand = ArucoParams {
                        adaptive_radius: r,
                        adaptive_c: c,
                        contrast,
                        gamma,
                        ..base
                    };
                    if cand == best {
                        continue;
                    }
                    let s = score_params(&sample, &cand);
                    evaluated += 1;
                    if s.is_better_than(&best_score) {
                        best_score = s;
                        best = cand;
                    }
                }
            }
        }
    }

    println!(
        "  evaluated {evaluated} combinations; best: {} markers across {}/{} images \
         (radius={}, c={}, contrast={}, gamma={})",
        best_score.total_markers,
        best_score.images_with_markers,
        sample.len(),
        best.adaptive_radius,
        best.adaptive_c,
        best.contrast,
        best.gamma
    );
    if best_score.total_markers == 0 {
        eprintln!(
            "warning: no markers detected under any tested parameters - check that the images \
             really contain markers from this dictionary (`--aruco-dict`)"
        );
    }
    Ok(best)
}

pub fn save(path: &Path, params: &ArucoParams) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_str = toml::to_string_pretty(params)?;
    std::fs::write(path, toml_str).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Option<ArucoParams>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(
            toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}
