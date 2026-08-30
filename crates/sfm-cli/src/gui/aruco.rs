//! Single-image ArUco tester: try the detector and its parameters on one real
//! frame, see what it found, and iterate.
//!
//! Detection parameters are unusually consequential and unusually hard to
//! judge from aggregate output. `--find-params` sweeps them across a sample of
//! the dataset and reports a score, which answers "which setting is best" but
//! not "why did this frame find nothing" - and a run over hundreds of
//! twelve-megapixel frames is a slow way to ask that question. This view
//! answers it on one frame in a third of a second, with the quads drawn on the
//! image so a miss is visible rather than inferred from a count.
//!
//! Detection runs on a worker thread. It is fast now, but "fast" is exactly
//! what this view exists to check, and a view that freezes while measuring
//! cannot show a regression that makes it slow again.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use sfm_features::aruco::ArucoParams;

/// One detected marker, as the overlay needs it.
pub struct Marker {
    pub id: u32,
    /// Corner positions in source-image pixels, in detection order.
    pub corners: [[f32; 2]; 4],
}

pub struct Detection {
    pub markers: Vec<Marker>,
    /// Data cells per side that the detector inferred for this frame.
    pub family: Option<usize>,
    pub num_corners: usize,
    pub decode: Duration,
    pub detect: Duration,
    pub source_size: [f32; 2],
}

#[derive(Default)]
pub enum State {
    #[default]
    Idle,
    Running,
    Done(Box<Detection>),
    Failed(String),
}

pub struct Tester {
    pub params: ArucoParams,
    /// Images offered for testing, discovered from the project.
    pub candidates: Vec<PathBuf>,
    pub selected: usize,
    pub state: Arc<Mutex<State>>,
    /// Cached texture of whichever image is being shown.
    pub texture: Option<(PathBuf, egui::TextureHandle, [f32; 2])>,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub show_ids: bool,
}

impl Default for Tester {
    fn default() -> Self {
        Tester {
            params: ArucoParams::default(),
            candidates: Vec::new(),
            selected: 0,
            state: Arc::new(Mutex::new(State::Idle)),
            texture: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            show_ids: true,
        }
    }
}

impl Tester {
    /// Runs detection on the selected image, off the UI thread.
    pub fn spawn(&self, ctx: &egui::Context) {
        let Some(path) = self.candidates.get(self.selected).cloned() else {
            return;
        };
        match self.state.lock() {
            Ok(mut s) if !matches!(*s, State::Running) => *s = State::Running,
            _ => return,
        }
        let slot = self.state.clone();
        let params = self.params;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<Detection, String> {
                let t0 = std::time::Instant::now();
                let img = image::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let decode = t0.elapsed();
                let t1 = std::time::Instant::now();
                let fs = sfm_features::aruco::detect(&img, &params);
                let detect = t1.elapsed();
                Ok(Detection {
                    family: sfm_features::aruco::detect_family(&img, &params),
                    markers: group_markers(&fs),
                    num_corners: fs.keypoints.len(),
                    decode,
                    detect,
                    source_size: [img.width() as f32, img.height() as f32],
                })
            })();
            if let Ok(mut s) = slot.lock() {
                *s = match result {
                    Ok(d) => State::Done(Box::new(d)),
                    Err(e) => State::Failed(e),
                };
            }
            ctx.request_repaint();
        });
    }
}

/// Rebuilds whole markers from the flat corner list the detector emits.
///
/// `FeatureSet` is deliberately a flat keypoint array - that is what matching
/// and reconstruction want - so the four corners of one marker are four
/// separate entries tagged with `(capture, marker id, corner index)` in the
/// descriptor blob. Drawing a quad needs them back together.
fn group_markers(fs: &sfm_core::FeatureSet) -> Vec<Marker> {
    let sfm_core::Descriptors::MarkerCorner { data } = &fs.descriptors else {
        return Vec::new();
    };
    const STRIDE: usize = 12;
    let mut by_marker: std::collections::BTreeMap<u32, [[f32; 2]; 4]> = Default::default();
    let mut seen: std::collections::BTreeMap<u32, u8> = Default::default();
    for (i, kp) in fs.keypoints.iter().enumerate() {
        let Some(rec) = data.get(i * STRIDE..i * STRIDE + STRIDE) else {
            continue;
        };
        let marker_id = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
        let corner = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as usize;
        if corner > 3 {
            continue;
        }
        by_marker.entry(marker_id).or_default()[corner] = [kp.x, kp.y];
        *seen.entry(marker_id).or_default() |= 1 << corner;
    }
    by_marker
        .into_iter()
        // Only quads with all four corners; a partial one would draw a
        // triangle through the origin and look like a detection bug.
        .filter(|(id, _)| seen.get(id).copied().unwrap_or(0) == 0b1111)
        .map(|(id, corners)| Marker { id, corners })
        .collect()
}

/// Distinct-ish colour per marker id, so neighbouring markers are separable.
pub fn marker_color(id: u32) -> egui::Color32 {
    // Golden-angle hue rotation: adjacent ids land far apart on the wheel.
    let h = (id as f32 * 137.508) % 360.0;
    let (r, g, b) = hsv(h, 0.75, 1.0);
    egui::Color32::from_rgb(r, g, b)
}

fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfm_core::{Descriptors, FeatureSet, Keypoint};

    fn kp(x: f32, y: f32) -> Keypoint {
        Keypoint {
            x,
            y,
            scale: 1.0,
            angle: 0.0,
            response: 1.0,
        }
    }

    fn rec(marker: u32, corner: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&marker.to_le_bytes());
        v.extend_from_slice(&corner.to_le_bytes());
        v
    }

    #[test]
    fn corners_regroup_into_whole_markers() {
        let mut keypoints = Vec::new();
        let mut data = Vec::new();
        for (marker, base) in [(7u32, 0.0f32), (3, 100.0)] {
            for corner in 0..4u32 {
                keypoints.push(kp(base + corner as f32, base));
                data.extend_from_slice(&rec(marker, corner));
            }
        }
        let fs = FeatureSet {
            keypoints,
            descriptors: Descriptors::MarkerCorner { data },
        };
        let markers = group_markers(&fs);
        assert_eq!(markers.len(), 2);
        // Ordered by id, and each corner lands in its own slot.
        assert_eq!(markers[0].id, 3);
        assert_eq!(markers[1].id, 7);
        assert_eq!(markers[1].corners[2], [2.0, 0.0]);
    }

    #[test]
    fn a_marker_missing_a_corner_is_dropped_not_drawn_through_the_origin() {
        let mut keypoints = Vec::new();
        let mut data = Vec::new();
        for corner in 0..3u32 {
            keypoints.push(kp(corner as f32, 0.0));
            data.extend_from_slice(&rec(9, corner));
        }
        let fs = FeatureSet {
            keypoints,
            descriptors: Descriptors::MarkerCorner { data },
        };
        assert!(group_markers(&fs).is_empty());
    }

    #[test]
    fn non_marker_descriptors_yield_nothing() {
        let fs = FeatureSet {
            keypoints: vec![kp(1.0, 1.0)],
            descriptors: Descriptors::Float32 {
                dim: 1,
                data: vec![0.0],
            },
        };
        assert!(group_markers(&fs).is_empty());
    }

    #[test]
    fn marker_colors_separate_adjacent_ids() {
        let a = marker_color(0);
        let b = marker_color(1);
        let d = (a.r() as i32 - b.r() as i32).abs()
            + (a.g() as i32 - b.g() as i32).abs()
            + (a.b() as i32 - b.b() as i32).abs();
        assert!(d > 120, "ids 0 and 1 look too similar ({a:?} vs {b:?})");
    }
}
