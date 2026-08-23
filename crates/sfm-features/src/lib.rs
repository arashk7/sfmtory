pub mod aruco;
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
}

impl DetectorKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sift" => Some(DetectorKind::Sift),
            "orb" => Some(DetectorKind::Orb),
            "aruco" => Some(DetectorKind::Aruco),
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
}

impl DetectorConfig {
    pub fn new(kind: DetectorKind) -> Self {
        let mut sift = sift::SiftParams::default();
        let mut orb = orb::OrbParams::default();
        DetectorConfig {
            kind,
            max_features: None,
            sift: {
                sift.max_features = None;
                sift
            },
            orb: {
                orb.max_features = None;
                orb
            },
            aruco: aruco::ArucoParams::default(),
        }
    }

    pub fn with_max_features(mut self, max: Option<usize>) -> Self {
        self.max_features = max;
        self.sift.max_features = max;
        self.orb.max_features = max;
        self
    }
}

/// Run the configured detector on one already-loaded image.
pub fn detect(img: &image::DynamicImage, config: &DetectorConfig) -> FeatureSet {
    match config.kind {
        DetectorKind::Sift => sift::detect(img, &config.sift),
        DetectorKind::Orb => orb::detect(img, &config.orb),
        DetectorKind::Aruco => aruco::detect(img, &config.aruco),
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
    Ok(detect(&img, config))
}
