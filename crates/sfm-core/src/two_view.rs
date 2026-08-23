//! Verified two-view match record, as stored in the project database by
//! `sfm match` and consumed by `sfm map`. Kept in `sfm-core` (rather than
//! `sfm-match`, which produces it) so the reconstruction crate can depend on
//! just the data shape without pulling in the matching/RANSAC machinery.

use serde::{Deserialize, Serialize};

use crate::pose::Pose;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoViewGeometryRecord {
    /// Relative pose of image 2 with image 1 fixed at the origin; translation
    /// is unit-scale (essential-matrix decomposition has no absolute scale).
    pub pose: Pose,
    /// Geometrically-verified `(keypoint_idx_in_image1, keypoint_idx_in_image2)`
    /// pairs - already filtered down from the raw descriptor-matcher output.
    pub inlier_matches: Vec<(u32, u32)>,
}
