pub mod aruco;
pub mod disk;
pub mod geom2d;
pub mod gray;
pub mod homography;
pub mod orb;
pub mod sift;

use sfm_core::FeatureSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorKind {
    Sift,
    Orb,
    Aruco,
    Disk,
}

impl DetectorKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sift" => Some(DetectorKind::Sift),
            "orb" => Some(DetectorKind::Orb),
            "aruco" => Some(DetectorKind::Aruco),
            "disk" => Some(DetectorKind::Disk),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub kind: DetectorKind,
    pub max_features: Option<usize>,
    pub sift: sift::SiftParams,
    pub orb: orb::OrbParams,
    pub aruco: aruco::ArucoParams,
    pub disk: disk::DiskParams,
}

impl DetectorConfig {
    /// Detector-specific `max_features` caps start at each detector's own
    /// sensible built-in default (e.g. SIFT's 8000, matching the same order
    /// of magnitude as COLMAP's own default `max_num_features=8192`) - not
    /// uncapped. An earlier version reset both to `None` here unconditionally
    /// (only a user-supplied `--max-features` could ever cap anything),
    /// which combined with other detector-accuracy fixes measurably
    /// ballooned real per-image feature counts to ~4-5x *more* than COLMAP's
    /// own default, not just matching it - purely more low-quality/marginal
    /// keypoints diluting the strong ones and slowing every downstream
    /// O(n^2) matching pair for no accuracy benefit (see PLAN.md's
    /// accuracy/density investigation). Call `with_max_features` to override
    /// this default explicitly (e.g. from a CLI flag).
    pub fn new(kind: DetectorKind) -> Self {
        DetectorConfig {
            kind,
            max_features: None,
            sift: sift::SiftParams::default(),
            orb: orb::OrbParams::default(),
            aruco: aruco::ArucoParams::default(),
            disk: disk::DiskParams::default(),
        }
    }

    /// Overrides every detector's `max_features` cap. Pass `None` to
    /// explicitly request *uncapped* extraction (not the default - see
    /// `new`'s doc comment for why unbounded isn't the baseline behavior).
    pub fn with_max_features(mut self, max: Option<usize>) -> Self {
        self.max_features = max;
        self.sift.max_features = max;
        self.orb.max_features = max;
        self.disk.max_features = max;
        self
    }
}

/// Run the configured detector on one already-loaded image. Fallible only
/// for `Disk` (model download/inference can fail); the classical detectors
/// can't fail on a well-formed in-memory image, so they're wrapped in `Ok`.
pub fn detect(img: &image::DynamicImage, config: &DetectorConfig) -> sfm_core::Result<FeatureSet> {
    match config.kind {
        DetectorKind::Sift => Ok(sift::detect(img, &config.sift)),
        DetectorKind::Orb => Ok(orb::detect(img, &config.orb)),
        DetectorKind::Aruco => Ok(aruco::detect(img, &config.aruco)),
        DetectorKind::Disk => disk::detect(img, &config.disk),
    }
}

/// Load an image from disk and run the configured detector on it.
pub fn detect_file(
    path: &std::path::Path,
    config: &DetectorConfig,
) -> sfm_core::Result<FeatureSet> {
    let img = image::open(path).map_err(|e| {
        sfm_core::SfmError::Other(format!("failed to load image {}: {e}", path.display()))
    })?;
    detect(&img, config)
}
