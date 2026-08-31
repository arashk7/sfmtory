//! Everything the diagnostic panels display, computed from a reconstruction
//! and kept free of any `egui` types so it can be unit-tested directly.
//!
//! The panels exist because the pipeline already *knows* why a hard dataset
//! failed and simply never says so. `data/iphone_marker` is the motivating
//! case: every track had exactly two observations, so a shared focal length
//! had no multi-view redundancy to be observed from, self-calibration was
//! correctly declined, and the tool went on reporting the unrefined 1536px
//! initial guess as though it were a result. Nothing was wrong except that
//! the reason was invisible. These structures make it sayable.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};
use sfm_core::{CameraModel, Reconstruction};

/// One reprojected observation, recomputed from the model's own geometry.
///
/// Deliberately *not* read from `Point3D::error`: that field records the mean
/// error over a point's track as of the last bundle adjustment that touched
/// it, so it is both an average (hiding which observation is bad) and
/// potentially stale with respect to the poses and intrinsics actually
/// written to disk. `sfm eval` recomputes for the same reason.
#[derive(Debug, Clone, Copy)]
pub struct Obs {
    pub point_id: u64,
    pub image_id: u32,
    /// Detected keypoint position, in pixels.
    pub measured: (f64, f64),
    /// Where the triangulated point actually lands, in pixels.
    pub projected: (f64, f64),
    pub residual_px: f64,
}

#[derive(Debug, Default, Clone)]
pub struct TrackStats {
    /// `(track length, number of points with that length)`, ascending.
    pub histogram: Vec<(usize, usize)>,
    pub min: usize,
    pub max: usize,
    pub mean: f64,
    pub median: f64,
    pub total_points: usize,
}

/// Why a camera's focal length does or does not differ from its initial guess.
///
/// The reconstruction writes only its final intrinsics, so the verdict is
/// reconstructed by comparing them against the initial guess still held in the
/// project database (`feature`/`init-cam` write it there; `map` never
/// overwrites it) and by re-testing the same eligibility gate the pipeline
/// applies. That avoids threading a report out through every bundle-adjustment
/// call site, at the cost of not being able to name *which* pass moved a focal
/// that did move - hence `Refined` says only that it moved.
#[derive(Debug, Clone, PartialEq)]
pub enum FocalVerdict {
    /// Intrinsics were held at a value the project config pinned.
    Pinned,
    /// The camera never met the images-per-camera gate, so refinement was
    /// never attempted for it.
    NotEligible { num_images: usize },
    /// Eligible, but the final focal is identical to the initial guess: the
    /// refined fit did not beat the fixed-intrinsics fit and was discarded.
    Rejected,
    /// The focal moved off its initial guess.
    Refined { relative_change: f64 },
    /// No initial guess available to compare against (no database, or the
    /// camera is absent from it).
    Unknown,
}

impl FocalVerdict {
    /// One-line summary for the panel header.
    pub fn headline(&self) -> String {
        match self {
            FocalVerdict::Pinned => "focal pinned by sfm.toml (refine = false)".into(),
            FocalVerdict::NotEligible { num_images } => format!(
                "focal NOT refined - only {num_images} image(s) on this camera, needs {}",
                sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
            ),
            FocalVerdict::Rejected => {
                "focal NOT refined - self-calibration ran and its result was rejected".into()
            }
            FocalVerdict::Refined { relative_change } => {
                format!(
                    "focal refined ({:+.2}% from the initial guess)",
                    relative_change * 100.0
                )
            }
            FocalVerdict::Unknown => "focal unknown - no initial guess to compare against".into(),
        }
    }

    /// Whether this should be drawn as a warning rather than as a result.
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            FocalVerdict::NotEligible { .. } | FocalVerdict::Rejected
        )
    }
}

#[derive(Debug, Clone)]
pub struct CameraDiag {
    pub camera_id: u32,
    pub model_name: String,
    pub num_images: usize,
    pub num_observations: usize,
    /// Focal length as written by `map`. Non-square pixels are summarised as
    /// `fx` here; the panel prints both.
    pub focal_final: (f64, f64),
    /// Focal length before reconstruction, from the project database.
    pub focal_initial: Option<(f64, f64)>,
    pub verdict: FocalVerdict,
    /// Longest track among the points this camera observes. A maximum of 2 is
    /// the specific pathology that makes a shared focal unobservable, so it is
    /// carried alongside the verdict rather than left to the global histogram.
    pub max_track_len: usize,
    /// `(horizontal, diagonal)` field of view in degrees.
    ///
    /// Carried because it, not the focal length, is what says whether a camera
    /// model can represent this lens: a single radial term stops coping past
    /// roughly 70 degrees diagonal, and a rectilinear projection stops being a
    /// good description at all past about 100.
    pub field_of_view_deg: Option<(f64, f64)>,
}

impl CameraDiag {
    /// The supporting sentence under `verdict.headline()`, when there is one.
    pub fn evidence(&self) -> Option<String> {
        if !self.verdict.is_warning() {
            return None;
        }
        if self.max_track_len <= 2 {
            return Some(format!(
                "every track on this camera has at most {} observations, so no 3D point \
                 is seen by enough views to constrain a shared focal length",
                self.max_track_len
            ));
        }
        None
    }
}

#[derive(Debug, Default, Clone)]
pub struct ImageDiag {
    pub image_id: u32,
    pub num_keypoints: usize,
    /// Keypoints linked to a triangulated point.
    pub num_observations: usize,
    pub mean_residual_px: f64,
    pub max_residual_px: f64,
}

/// Reprojection residuals over the whole model, plus the indices needed to get
/// at one image's or one point's own observations without rescanning.
#[derive(Debug, Default, Clone)]
pub struct Residuals {
    pub all: Vec<Obs>,
    pub by_image: BTreeMap<u32, Vec<usize>>,
    pub by_point: BTreeMap<u64, Vec<usize>>,
    /// Mean residual per point - what the 3D view colours by.
    pub point_mean: BTreeMap<u64, f64>,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
    /// Observations whose point falls behind the camera; excluded from the
    /// statistics above because a reprojection is undefined for them.
    pub num_behind_camera: usize,
}

/// A best-fit plane through the point cloud, and how each camera views it.
///
/// Only meaningful for a planar target. Self-calibration from a single plane
/// is Zhang's method, and Zhang's method needs the plane tilted substantially
/// between shots: with every view at a similar angle, focal length trades off
/// against plane pose and many combinations reproject equally well. That is
/// exactly why `data/iphone_marker` cannot be calibrated from, and it is
/// invisible in any of the numbers the pipeline currently prints.
#[derive(Debug, Clone)]
pub struct PlaneDiag {
    pub centroid: Vector3<f64>,
    pub normal: Vector3<f64>,
    /// In-plane basis, so a view direction can be given an azimuth.
    pub basis: [Vector3<f64>; 2],
    /// Out-of-plane extent divided by the smaller in-plane extent. Near zero
    /// means the cloud really is a plane; approaching 1 means it is not.
    pub flatness: f64,
    pub views: Vec<ViewAngle>,
    pub tilt_min_deg: f64,
    pub tilt_max_deg: f64,
    /// How many of the 8 azimuth sectors contain a meaningfully tilted view.
    /// Reported because it is what the coverage plot draws, but it is *not*
    /// what the verdict tests - see `axis_spread_deg`.
    pub azimuth_sectors_covered: usize,
    /// Circular spread of the *rotation axis* the target is tilted about,
    /// in degrees, over the meaningfully-tilted views.
    ///
    /// This, not the tilt direction, is the quantity Zhang's method is
    /// degenerate in: rotating the board about a fixed axis in every view
    /// leaves the intrinsics unrecoverable however wide the tilt range is.
    /// A tilt direction and its opposite are the *same* axis - the board
    /// leaning toward you and away from you both rotate about the horizontal -
    /// so the azimuth is folded modulo 180 degrees before this is computed,
    /// and a plain count of tilt directions cannot substitute for it.
    pub axis_spread_deg: f64,
    pub verdict: CoverageVerdict,
}

#[derive(Debug, Clone, Copy)]
pub struct ViewAngle {
    pub image_id: u32,
    /// Angle between the optical axis and the plane normal: 0 deg is
    /// fronto-parallel, 90 deg is grazing.
    pub tilt_deg: f64,
    /// Which way the view is tilted, in the plane's own basis, 0..360.
    /// Meaningless at near-zero tilt, where the in-plane component vanishes.
    pub azimuth_deg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoverageVerdict {
    /// The cloud is not planar, so plane-based calibration diversity is not
    /// the relevant question for this dataset.
    NotPlanar,
    /// Enough tilt, spread over enough directions.
    Adequate,
    Narrow(String),
}

/// Below this the point cloud is treated as a plane. Chosen from the measured
/// datasets rather than as a round number: `data/iphone_marker`'s marker grid
/// on a laptop screen fits a plane to 0.114, while the three genuinely
/// three-dimensional datasets sit at 0.666 (`sceaux_castle`), 0.718
/// (`temple_ring`) and 0.719 (`temple_sparse_ring`). 0.25 leaves a factor of
/// two of margin on both sides of a gap that is, in practice, wide.
pub const PLANAR_FLATNESS: f64 = 0.25;

/// A view has to be tilted by at least this much before its azimuth means
/// anything - below it the optical axis is nearly along the normal and the
/// in-plane component is dominated by noise.
const MIN_MEANINGFUL_TILT_DEG: f64 = 15.0;

/// Zhang's method wants the board seen from substantially different angles.
/// These are the thresholds the `Narrow` verdict fires below.
const MIN_TILT_RANGE_DEG: f64 = 20.0;

/// Minimum circular spread of the tilt axis, in degrees.
///
/// Calibrated against the real datasets rather than picked as a round number.
/// `data/iphone_marker` measures 8.0 degrees: of its ten meaningfully-tilted
/// photos, nine tilt the marker board about axes between 14 and 27 degrees and
/// the tenth about 174 (which is the same axis, the other way). So despite a
/// 36-degree tilt range and three occupied azimuth sectors it is a single-axis
/// capture, and the focal length it produces should not be trusted. The
/// threshold corresponds to roughly two equally-weighted orientations 30
/// degrees apart - the least that can be called two distinct axes; a
/// comfortable capture at 45 degrees apart measures ~24 degrees of spread.
const MIN_AXIS_SPREAD_DEG: f64 = 15.0;

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub tracks: TrackStats,
    pub cameras: Vec<CameraDiag>,
    pub images: BTreeMap<u32, ImageDiag>,
    pub residuals: Residuals,
    pub plane: Option<PlaneDiag>,
}

impl Diagnostics {
    /// `initial_cameras` comes from the project database (the pre-reconstruction
    /// guess); `pinned` are camera ids the config declared `refine = false`.
    /// Both are optional context - the rest is computed from `recon` alone.
    pub fn compute(
        recon: &Reconstruction,
        initial_cameras: &BTreeMap<u32, CameraModel>,
        pinned: &std::collections::BTreeSet<u32>,
    ) -> Self {
        let residuals = compute_residuals(recon);
        let tracks = track_stats(recon);

        let mut images: BTreeMap<u32, ImageDiag> = BTreeMap::new();
        for im in recon.images.values() {
            let idx = residuals.by_image.get(&im.id);
            let (mut sum, mut max, mut n) = (0.0, 0.0f64, 0usize);
            if let Some(idx) = idx {
                for &i in idx {
                    let r = residuals.all[i].residual_px;
                    sum += r;
                    max = max.max(r);
                    n += 1;
                }
            }
            images.insert(
                im.id,
                ImageDiag {
                    image_id: im.id,
                    num_keypoints: im.keypoints.len(),
                    num_observations: im.point3d_ids.iter().filter(|p| p.is_some()).count(),
                    mean_residual_px: if n > 0 { sum / n as f64 } else { 0.0 },
                    max_residual_px: max,
                },
            );
        }

        let cameras = camera_diags(recon, &residuals, initial_cameras, pinned);
        let plane = fit_plane(recon);

        Diagnostics {
            tracks,
            cameras,
            images,
            residuals,
            plane,
        }
    }
}

fn track_stats(recon: &Reconstruction) -> TrackStats {
    let mut lengths: Vec<usize> = recon.points3d.values().map(|p| p.track.len()).collect();
    if lengths.is_empty() {
        return TrackStats::default();
    }
    lengths.sort_unstable();
    let mut histogram: Vec<(usize, usize)> = Vec::new();
    for &l in &lengths {
        match histogram.last_mut() {
            Some((len, count)) if *len == l => *count += 1,
            _ => histogram.push((l, 1)),
        }
    }
    TrackStats {
        min: lengths[0],
        max: *lengths.last().unwrap(),
        mean: lengths.iter().sum::<usize>() as f64 / lengths.len() as f64,
        median: lengths[lengths.len() / 2] as f64,
        total_points: lengths.len(),
        histogram,
    }
}

/// Reprojects every observation through the model's own poses and intrinsics.
/// Mirrors `recompute_reprojection` in `main.rs` (`sfm eval`), but keeps the
/// per-observation detail the CLI summary throws away.
pub fn compute_residuals(recon: &Reconstruction) -> Residuals {
    let mut r = Residuals::default();
    for point in recon.points3d.values() {
        let mut sum = 0.0;
        let mut n = 0usize;
        for t in &point.track {
            let Some(image) = recon.images.get(&t.image_id) else {
                continue;
            };
            let Some(camera) = recon.cameras.get(&image.camera_id) else {
                continue;
            };
            let Some(&(u, v)) = image.keypoints.get(t.point2d_idx as usize) else {
                continue;
            };
            let pc = image.pose.transform_point(&point.xyz);
            if pc.z <= 1e-9 {
                r.num_behind_camera += 1;
                continue;
            }
            let (px, py) = camera.model.project(&pc);
            let residual_px = ((px - u as f64).powi(2) + (py - v as f64).powi(2)).sqrt();
            let idx = r.all.len();
            r.all.push(Obs {
                point_id: point.id,
                image_id: t.image_id,
                measured: (u as f64, v as f64),
                projected: (px, py),
                residual_px,
            });
            r.by_image.entry(t.image_id).or_default().push(idx);
            r.by_point.entry(point.id).or_default().push(idx);
            sum += residual_px;
            n += 1;
        }
        if n > 0 {
            r.point_mean.insert(point.id, sum / n as f64);
        }
    }

    let mut sorted: Vec<f64> = r.all.iter().map(|o| o.residual_px).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| -> f64 {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((sorted.len() - 1) as f64 * q).round() as usize]
        }
    };
    r.mean = if sorted.is_empty() {
        0.0
    } else {
        sorted.iter().sum::<f64>() / sorted.len() as f64
    };
    r.median = pick(0.5);
    r.p95 = pick(0.95);
    r.max = sorted.last().copied().unwrap_or(0.0);
    r
}

fn camera_diags(
    recon: &Reconstruction,
    residuals: &Residuals,
    initial_cameras: &BTreeMap<u32, CameraModel>,
    pinned: &std::collections::BTreeSet<u32>,
) -> Vec<CameraDiag> {
    let mut per_camera_images: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for im in recon.images.values() {
        per_camera_images
            .entry(im.camera_id)
            .or_default()
            .push(im.id);
    }
    // Which camera each image belongs to, so a point's track can be attributed.
    let camera_of_image: BTreeMap<u32, u32> = recon
        .images
        .values()
        .map(|im| (im.id, im.camera_id))
        .collect();
    let mut max_track: BTreeMap<u32, usize> = BTreeMap::new();
    for p in recon.points3d.values() {
        for t in &p.track {
            if let Some(cam) = camera_of_image.get(&t.image_id) {
                let e = max_track.entry(*cam).or_insert(0);
                *e = (*e).max(p.track.len());
            }
        }
    }

    recon
        .cameras
        .values()
        .map(|cam| {
            let image_ids = per_camera_images.get(&cam.camera_id);
            let num_images = image_ids.map(|v| v.len()).unwrap_or(0);
            let num_observations = image_ids
                .map(|ids| {
                    ids.iter()
                        .map(|id| residuals.by_image.get(id).map(Vec::len).unwrap_or(0))
                        .sum()
                })
                .unwrap_or(0);
            let focal_final = cam.model.focal_lengths();
            let focal_initial = initial_cameras
                .get(&cam.camera_id)
                .map(|m| m.focal_lengths());
            let verdict = focal_verdict(
                focal_initial,
                focal_final,
                num_images,
                pinned.contains(&cam.camera_id),
            );
            let (fx, _) = focal_final;
            let field_of_view_deg = (fx > 1.0).then(|| {
                let (w, h) = (cam.width as f64, cam.height as f64);
                (
                    2.0 * (w / (2.0 * fx)).atan().to_degrees(),
                    2.0 * (w.hypot(h) / (2.0 * fx)).atan().to_degrees(),
                )
            });
            CameraDiag {
                camera_id: cam.camera_id,
                model_name: cam.model.name().to_string(),
                num_images,
                num_observations,
                focal_final,
                focal_initial,
                verdict,
                max_track_len: max_track.get(&cam.camera_id).copied().unwrap_or(0),
                field_of_view_deg,
            }
        })
        .collect()
}

/// The verdict logic, split out so it can be tested against each case without
/// building a whole reconstruction.
pub fn focal_verdict(
    initial: Option<(f64, f64)>,
    final_: (f64, f64),
    num_images: usize,
    pinned: bool,
) -> FocalVerdict {
    if pinned {
        return FocalVerdict::Pinned;
    }
    let Some(initial) = initial else {
        return FocalVerdict::Unknown;
    };
    // Bundle adjustment writes a refined focal back in full precision, so an
    // exactly-preserved value means no refinement was ever accepted rather
    // than one that happened to converge back to the guess. The tolerance is
    // for the text round-trip through `cameras.txt`, not for judgement.
    let moved = (final_.0 - initial.0).abs() > initial.0.abs() * 1e-9
        || (final_.1 - initial.1).abs() > initial.1.abs() * 1e-9;
    if moved {
        let relative_change = if initial.0.abs() > 0.0 {
            (final_.0 - initial.0) / initial.0
        } else {
            0.0
        };
        return FocalVerdict::Refined { relative_change };
    }
    if num_images < sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS {
        FocalVerdict::NotEligible { num_images }
    } else {
        FocalVerdict::Rejected
    }
}

/// Best-fit plane through the points (PCA), plus each camera's angle to it.
/// Returns `None` when there is not enough structure to fit anything.
pub fn fit_plane(recon: &Reconstruction) -> Option<PlaneDiag> {
    let pts: Vec<Vector3<f64>> = recon.points3d.values().map(|p| p.xyz).collect();
    if pts.len() < 3 || recon.images.is_empty() {
        return None;
    }
    let centroid = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
    let mut cov = Matrix3::zeros();
    for p in &pts {
        let d = p - centroid;
        cov += d * d.transpose();
    }
    cov /= pts.len() as f64;

    let eig = cov.symmetric_eigen();
    // Ascending by eigenvalue: the smallest is the plane normal, the largest
    // the dominant in-plane direction.
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sd = |i: usize| eig.eigenvalues[order[i]].max(0.0).sqrt();
    let normal = eig.eigenvectors.column(order[0]).into_owned().normalize();
    let basis = [
        eig.eigenvectors.column(order[1]).into_owned().normalize(),
        eig.eigenvectors.column(order[2]).into_owned().normalize(),
    ];
    // Against the *smaller* in-plane extent, so a cloud that is long and thin
    // but genuinely flat is not mistaken for a thick one.
    let flatness = if sd(1) > 1e-12 { sd(0) / sd(1) } else { 1.0 };

    let mut views = Vec::new();
    for im in recon.images.values() {
        // Optical axis in world coordinates: the camera looks along +z in its
        // own frame, and `pose.rotation` is world-to-camera.
        let r_cw = im
            .pose
            .rotation
            .to_rotation_matrix()
            .into_inner()
            .transpose();
        let dir = r_cw * Vector3::new(0.0, 0.0, 1.0);
        let cos = (dir.dot(&normal)).clamp(-1.0, 1.0).abs();
        let tilt_deg = cos.acos().to_degrees();
        let in_plane = dir - normal * dir.dot(&normal);
        let azimuth_deg = if in_plane.norm() < 1e-9 {
            0.0
        } else {
            let a = in_plane.dot(&basis[1]).atan2(in_plane.dot(&basis[0]));
            (a.to_degrees() + 360.0) % 360.0
        };
        views.push(ViewAngle {
            image_id: im.id,
            tilt_deg,
            azimuth_deg,
        });
    }

    let tilt_min_deg = views.iter().map(|v| v.tilt_deg).fold(f64::MAX, f64::min);
    let tilt_max_deg = views.iter().map(|v| v.tilt_deg).fold(f64::MIN, f64::max);
    let sectors = azimuth_sectors(&views);
    let azimuth_sectors_covered = sectors.iter().filter(|&&n| n > 0).count();
    let axis_spread_deg = axis_spread_deg(&views);

    let verdict = if flatness > PLANAR_FLATNESS {
        CoverageVerdict::NotPlanar
    } else if tilt_max_deg - tilt_min_deg < MIN_TILT_RANGE_DEG {
        CoverageVerdict::Narrow(format!(
            "every view is within {:.1} deg of the same angle to the target plane \
             ({:.1}-{:.1} deg). Zhang's method needs the board tilted substantially \
             between shots, so the focal length is close to unidentifiable here.",
            tilt_max_deg - tilt_min_deg,
            tilt_min_deg,
            tilt_max_deg
        ))
    } else if axis_spread_deg < MIN_AXIS_SPREAD_DEG {
        CoverageVerdict::Narrow(format!(
            "the target is tilted by a healthy {:.1}-{:.1} deg, but always about \
             nearly the same axis (spread {axis_spread_deg:.1} deg, want at least \
             {MIN_AXIS_SPREAD_DEG:.0} deg). Rotating the board about one fixed axis \
             is a degenerate configuration for single-plane self-calibration however \
             wide the tilt range is: tilt it about a second, roughly perpendicular \
             axis as well.",
            tilt_min_deg, tilt_max_deg
        ))
    } else {
        CoverageVerdict::Adequate
    };

    Some(PlaneDiag {
        centroid,
        normal,
        basis,
        flatness,
        views,
        tilt_min_deg,
        tilt_max_deg,
        azimuth_sectors_covered,
        axis_spread_deg,
        verdict,
    })
}

/// Circular standard deviation of the tilt axis, in degrees.
///
/// The tilt axis is an undirected line in the plane, so this is *axial* rather
/// than directional data: azimuths of 20 and 200 degrees describe the same
/// axis. The standard treatment is to double the angles, take the ordinary
/// circular mean resultant, and halve the resulting spread - which is what
/// this does. Returns the maximum spread when there are too few tilted views
/// to say anything, so a capture with nothing to measure is never reported as
/// degenerate on this criterion.
pub fn axis_spread_deg(views: &[ViewAngle]) -> f64 {
    /// Perfectly uniform axes give a resultant of zero and an unbounded
    /// spread, so cap at the value a uniform distribution would imply.
    const MAX_SPREAD_DEG: f64 = 45.0;
    let tilted: Vec<f64> = views
        .iter()
        .filter(|v| v.tilt_deg >= MIN_MEANINGFUL_TILT_DEG)
        .map(|v| v.azimuth_deg.to_radians())
        .collect();
    if tilted.len() < 2 {
        return MAX_SPREAD_DEG;
    }
    let (mut cs, mut sn) = (0.0, 0.0);
    for a in &tilted {
        cs += (2.0 * a).cos();
        sn += (2.0 * a).sin();
    }
    let n = tilted.len() as f64;
    let r = ((cs / n).powi(2) + (sn / n).powi(2))
        .sqrt()
        .clamp(1e-12, 1.0);
    ((-2.0 * r.ln()).sqrt() * 0.5)
        .to_degrees()
        .min(MAX_SPREAD_DEG)
}

/// Counts of meaningfully-tilted views per 45-degree azimuth sector. Also the
/// basis of the capture-coverage view, which shows the empty sectors as the
/// orientations still to shoot.
pub fn azimuth_sectors(views: &[ViewAngle]) -> [usize; 8] {
    let mut sectors = [0usize; 8];
    for v in views {
        if v.tilt_deg < MIN_MEANINGFUL_TILT_DEG {
            continue;
        }
        let s = ((v.azimuth_deg / 45.0).floor() as usize).min(7);
        sectors[s] += 1;
    }
    sectors
}

/// Tilt bands used by the coverage view, in degrees. The innermost band is the
/// fronto-parallel one, whose azimuth carries no information.
pub const TILT_BANDS: [(f64, f64); 4] = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 90.0)];

/// `[tilt band][azimuth sector]` occupancy. The fronto-parallel band is
/// collapsed into sector 0, since azimuth is undefined there.
pub fn coverage_grid(views: &[ViewAngle]) -> [[usize; 8]; 4] {
    let mut grid = [[0usize; 8]; 4];
    for v in views {
        let band = TILT_BANDS
            .iter()
            .position(|&(lo, hi)| v.tilt_deg >= lo && v.tilt_deg < hi)
            .unwrap_or(TILT_BANDS.len() - 1);
        let sector = if band == 0 {
            0
        } else {
            ((v.azimuth_deg / 45.0).floor() as usize).min(7)
        };
        grid[band][sector] += 1;
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;
    use sfm_core::{Camera, Image, Point3D, Pose, TrackElement};

    fn pinhole(f: f64) -> CameraModel {
        CameraModel::Pinhole {
            fx: f,
            fy: f,
            cx: 320.0,
            cy: 240.0,
        }
    }

    /// A camera at `center` looking at the origin, with `f` as its focal.
    fn looking_at(id: u32, center: Vector3<f64>) -> Image {
        let fwd = (-center).normalize();
        let up_ref = if fwd.y.abs() > 0.999 {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let right = fwd.cross(&up_ref).normalize();
        // Negated so the triple is right-handed (x right, y *down*, z forward -
        // the computer-vision convention). Rows are the camera axes, making
        // this world-to-camera.
        let down = -right.cross(&fwd);
        let r_wc = Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd.transpose()]);
        let rot =
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(r_wc));
        Image {
            id,
            camera_id: 1,
            name: format!("img{id}.jpg"),
            pose: Pose {
                rotation: rot,
                translation: -(r_wc * center),
            },
            keypoints: Vec::new(),
            point3d_ids: Vec::new(),
        }
    }

    /// Points on the z = 0 plane, plus cameras placed at the given
    /// (tilt, azimuth) pairs relative to that plane's normal (+z).
    fn planar_scene(angles: &[(f64, f64)]) -> Reconstruction {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: pinhole(500.0),
                width: 640,
                height: 480,
            },
        );
        let mut id = 0u64;
        for i in 0..5 {
            for j in 0..5 {
                id += 1;
                recon.points3d.insert(
                    id,
                    Point3D {
                        id,
                        xyz: Vector3::new(i as f64 - 2.0, j as f64 - 2.0, 0.0),
                        color: [0, 0, 0],
                        error: 0.0,
                        track: Vec::new(),
                    },
                );
            }
        }
        for (k, &(tilt, az)) in angles.iter().enumerate() {
            let (t, a) = (tilt.to_radians(), az.to_radians());
            let center = Vector3::new(t.sin() * a.cos(), t.sin() * a.sin(), t.cos()) * 10.0;
            let im = looking_at(k as u32 + 1, center);
            recon.images.insert(im.id, im);
        }
        recon
    }

    #[test]
    fn track_histogram_counts_each_length() {
        let mut recon = Reconstruction::new();
        for (id, len) in [(1u64, 2usize), (2, 2), (3, 5), (4, 3)] {
            recon.points3d.insert(
                id,
                Point3D {
                    id,
                    xyz: Vector3::zeros(),
                    color: [0, 0, 0],
                    error: 0.0,
                    track: (0..len)
                        .map(|i| TrackElement {
                            image_id: i as u32,
                            point2d_idx: 0,
                        })
                        .collect(),
                },
            );
        }
        let s = track_stats(&recon);
        assert_eq!(s.histogram, vec![(2, 2), (3, 1), (5, 1)]);
        assert_eq!(s.min, 2);
        assert_eq!(s.max, 5);
        assert_eq!(s.total_points, 4);
        assert_eq!(s.median, 3.0);
    }

    #[test]
    fn verdict_separates_never_attempted_from_rejected() {
        // Unchanged focal with too few images: never eligible.
        assert_eq!(
            focal_verdict(Some((1536.0, 1536.0)), (1536.0, 1536.0), 3, false),
            FocalVerdict::NotEligible { num_images: 3 }
        );
        // Unchanged focal with enough images: it ran and lost the comparison.
        assert_eq!(
            focal_verdict(Some((1536.0, 1536.0)), (1536.0, 1536.0), 13, false),
            FocalVerdict::Rejected
        );
        // Moved: refined, and the sign of the change is reported.
        match focal_verdict(Some((1000.0, 1000.0)), (1100.0, 1100.0), 13, false) {
            FocalVerdict::Refined { relative_change } => {
                assert!((relative_change - 0.1).abs() < 1e-12);
            }
            other => panic!("expected Refined, got {other:?}"),
        }
        // A pinned camera is never a warning, however little it moved.
        assert_eq!(
            focal_verdict(Some((1536.0, 1536.0)), (1536.0, 1536.0), 3, true),
            FocalVerdict::Pinned
        );
        assert_eq!(
            focal_verdict(None, (1536.0, 1536.0), 13, false),
            FocalVerdict::Unknown
        );
    }

    #[test]
    fn residuals_are_recomputed_not_read_from_the_error_field() {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: pinhole(500.0),
                width: 640,
                height: 480,
            },
        );
        let mut im = looking_at(1, Vector3::new(0.0, 0.0, -10.0));
        // Place the keypoint 3px away from where the point actually projects.
        let p = Vector3::new(0.0, 0.0, 0.0);
        let pc = im.pose.transform_point(&p);
        let (px, py) = recon.cameras[&1].model.project(&pc);
        im.keypoints.push(((px + 3.0) as f32, py as f32));
        im.point3d_ids.push(Some(1));
        recon.images.insert(1, im);
        recon.points3d.insert(
            1,
            Point3D {
                id: 1,
                xyz: p,
                color: [0, 0, 0],
                // A deliberately wrong stored error, which must be ignored.
                error: 999.0,
                track: vec![TrackElement {
                    image_id: 1,
                    point2d_idx: 0,
                }],
            },
        );
        let r = compute_residuals(&recon);
        assert_eq!(r.all.len(), 1);
        assert!((r.all[0].residual_px - 3.0).abs() < 1e-3, "{:?}", r.all[0]);
        assert!((r.point_mean[&1] - 3.0).abs() < 1e-3);
        assert_eq!(r.num_behind_camera, 0);
    }

    #[test]
    fn plane_fit_recovers_a_known_normal_and_flags_narrow_tilt() {
        // Every camera within 5 degrees of fronto-parallel: the iphone_marker
        // failure mode.
        let recon = planar_scene(&[(0.0, 0.0), (3.0, 10.0), (5.0, 200.0), (2.0, 90.0)]);
        let plane = fit_plane(&recon).expect("planar scene fits");
        assert!(plane.flatness < 1e-6, "flatness {}", plane.flatness);
        assert!(plane.normal.z.abs() > 0.999, "normal {:?}", plane.normal);
        assert!(plane.tilt_max_deg < 6.0);
        match &plane.verdict {
            CoverageVerdict::Narrow(msg) => assert!(msg.contains("same angle")),
            other => panic!("expected Narrow, got {other:?}"),
        }
    }

    #[test]
    fn plane_fit_accepts_a_well_tilted_capture() {
        // Tilted about several genuinely different axes: 20, 110 and 155
        // degrees are distinct lines, not one line seen from both ends.
        let recon = planar_scene(&[
            (5.0, 0.0),
            (30.0, 20.0),
            (35.0, 110.0),
            (40.0, 155.0),
            (25.0, 65.0),
        ]);
        let plane = fit_plane(&recon).unwrap();
        assert!(plane.tilt_max_deg - plane.tilt_min_deg > 20.0);
        assert!(
            plane.axis_spread_deg >= MIN_AXIS_SPREAD_DEG,
            "axis spread was {}",
            plane.axis_spread_deg
        );
        assert_eq!(plane.verdict, CoverageVerdict::Adequate);
    }

    #[test]
    fn tilt_about_a_single_axis_is_narrow_even_with_a_wide_range() {
        // Wide tilt range, but every tilt in the same direction.
        let recon = planar_scene(&[(2.0, 0.0), (20.0, 10.0), (35.0, 12.0), (45.0, 8.0)]);
        let plane = fit_plane(&recon).unwrap();
        assert!(plane.tilt_max_deg - plane.tilt_min_deg > 20.0);
        assert!(plane.axis_spread_deg < MIN_AXIS_SPREAD_DEG);
        match &plane.verdict {
            CoverageVerdict::Narrow(msg) => assert!(msg.contains("same axis")),
            other => panic!("expected Narrow, got {other:?}"),
        }
    }

    #[test]
    fn opposing_tilt_directions_are_one_axis_not_two() {
        // The `data/iphone_marker` shape: a wide tilt range spread over two
        // *opposite* azimuths, which is one rotation axis, not two. Counting
        // azimuth sectors calls this well-covered; counting axes does not, and
        // the axis is what Zhang's method is degenerate in.
        let recon = planar_scene(&[
            (2.0, 130.0),
            (16.0, 14.0),
            (20.0, 20.0),
            (24.0, 201.0),
            (30.0, 199.0),
            (33.0, 199.0),
            (38.0, 200.0),
        ]);
        let plane = fit_plane(&recon).unwrap();
        assert!(plane.tilt_max_deg - plane.tilt_min_deg > MIN_TILT_RANGE_DEG);
        assert!(
            plane.azimuth_sectors_covered >= 2,
            "the opposing directions do occupy separate sectors"
        );
        assert!(
            plane.axis_spread_deg < 10.0,
            "axis spread was {}",
            plane.axis_spread_deg
        );
        match &plane.verdict {
            CoverageVerdict::Narrow(msg) => assert!(msg.contains("same axis")),
            other => panic!("expected Narrow, got {other:?}"),
        }
    }

    #[test]
    fn axis_spread_is_maximal_for_perpendicular_axes_and_undefined_for_too_few() {
        let v = |tilt: f64, az: f64| ViewAngle {
            image_id: 0,
            tilt_deg: tilt,
            azimuth_deg: az,
        };
        // Two perpendicular axes: as spread as axial data gets.
        assert!(axis_spread_deg(&[v(30.0, 0.0), v(30.0, 90.0)]) >= 44.9);
        // Opposite directions are one axis, so no spread at all.
        assert!(axis_spread_deg(&[v(30.0, 20.0), v(30.0, 200.0)]) < 1e-3);
        // Nothing meaningfully tilted: report maximum rather than degenerate,
        // so the tilt-range criterion is the one that speaks.
        assert_eq!(axis_spread_deg(&[v(3.0, 0.0), v(5.0, 90.0)]), 45.0);
    }

    #[test]
    fn a_non_planar_cloud_is_reported_as_such() {
        let mut recon = planar_scene(&[(10.0, 0.0), (30.0, 90.0)]);
        // Bend the grid into a saddle. Displacing along a *single* index would
        // only tilt the plane, which still fits one perfectly; a saddle has
        // genuine thickness in every direction.
        for p in recon.points3d.values_mut() {
            p.xyz.z = p.xyz.x * p.xyz.y * 0.5;
        }
        let plane = fit_plane(&recon).unwrap();
        assert!(plane.flatness > PLANAR_FLATNESS);
        assert_eq!(plane.verdict, CoverageVerdict::NotPlanar);
    }

    #[test]
    fn coverage_grid_collapses_the_fronto_parallel_band() {
        let views = vec![
            ViewAngle {
                image_id: 1,
                tilt_deg: 5.0,
                azimuth_deg: 200.0,
            },
            ViewAngle {
                image_id: 2,
                tilt_deg: 20.0,
                azimuth_deg: 10.0,
            },
            ViewAngle {
                image_id: 3,
                tilt_deg: 50.0,
                azimuth_deg: 190.0,
            },
        ];
        let grid = coverage_grid(&views);
        // Band 0 ignores azimuth entirely.
        assert_eq!(grid[0][0], 1);
        assert_eq!(grid[0].iter().sum::<usize>(), 1);
        assert_eq!(grid[1][0], 1);
        assert_eq!(grid[3][4], 1);
        assert_eq!(azimuth_sectors(&views), [1, 0, 0, 0, 1, 0, 0, 0]);
    }
}

#[cfg(test)]
mod fov_tests {
    use super::*;
    use sfm_core::Camera;

    /// The two focal estimates measured on the `scan2` rig, which differ by
    /// 2.1x and land on opposite sides of every model-choice threshold.
    #[test]
    fn field_of_view_separates_the_two_focal_estimates() {
        let fov = |f: f64| {
            let (w, h) = (3840.0f64, 3104.0f64);
            (
                2.0 * (w / (2.0 * f)).atan().to_degrees(),
                2.0 * (w.hypot(h) / (2.0 * f)).atan().to_degrees(),
            )
        };
        // The 1.2 x width placeholder: narrow, and a single radial term would
        // look adequate.
        let (h_placeholder, d_placeholder) = fov(4608.0);
        assert!((h_placeholder - 45.2).abs() < 0.2, "{h_placeholder}");
        assert!(d_placeholder < 70.0);

        // What init-cam actually measured from the marker squares: wide enough
        // that SIMPLE_RADIAL is the wrong model and fisheye is worth testing.
        let (h_measured, d_measured) = fov(2185.71);
        assert!((h_measured - 82.6).abs() < 0.2, "{h_measured}");
        assert!(d_measured > 95.0, "{d_measured}");
    }

    #[test]
    fn a_degenerate_focal_reports_no_field_of_view() {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: CameraModel::Pinhole {
                    fx: 0.0,
                    fy: 0.0,
                    cx: 0.0,
                    cy: 0.0,
                },
                width: 640,
                height: 480,
            },
        );
        let d = Diagnostics::compute(&recon, &BTreeMap::new(), &std::collections::BTreeSet::new());
        assert_eq!(d.cameras[0].field_of_view_deg, None);
    }
}
