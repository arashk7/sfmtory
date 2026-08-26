pub mod descriptor_match;
pub mod pairing;
pub mod vocab;

use sfm_core::{CameraModel, FeatureSet, TwoViewGeometryRecord};
use sfm_geometry::{estimate_two_view_geometry, to_normalized};

pub use descriptor_match::{match_descriptors, MatchParams};
pub use pairing::{exhaustive_pairs, sequential_pairs};
pub use vocab::{vocab_tree_pairs, VocabParams};

#[derive(Debug, Clone, Copy)]
pub struct VerificationParams {
    pub match_params: MatchParams,
    /// RANSAC inlier threshold, in pixels.
    pub ransac_threshold_px: f64,
    pub ransac_max_iterations: usize,
    /// A pair is rejected outright (kept out of the reconstruction) unless
    /// it clears both of these - the main defense against a coincidentally-
    /// matched but geometrically-nonsensical pair polluting the reconstruction.
    ///
    /// `min_inlier_ratio` in particular needs to stay low: architecture
    /// (repeated windows, columns, symmetric facade details - exactly what
    /// `data/sceaux_castle` is) generates large numbers of confident-looking
    /// but geometrically-wrong SIFT matches between similar-but-different
    /// repeated elements, which drags the *ratio* down even when the
    /// absolute inlier count is high and the pose is solid. A real pair with
    /// 83-133 correct RANSAC inliers got rejected here at the previous
    /// default of 0.25 purely on ratio, costing real registrations - see
    /// PLAN.md's real-data-testing entry. Now matches COLMAP's own default
    /// exactly (`TwoViewGeometryOptions::min_inlier_ratio = 0.0`, confirmed
    /// via `pycolmap`): no ratio gate at all, `min_inliers` alone (also
    /// COLMAP's own default value, 15) is the entire defense. An even lower
    /// non-zero floor was tried as a "sanity backstop" against a degenerate
    /// handful-of-coincidental-matches case, but measurably cost verified
    /// pairs relative to COLMAP on real data without evidence it ever
    /// caught a real bad pair - removed rather than kept on spec.
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
}

impl Default for VerificationParams {
    fn default() -> Self {
        VerificationParams {
            match_params: MatchParams::default(),
            ransac_threshold_px: 4.0,
            ransac_max_iterations: 2000,
            min_inliers: 15,
            min_inlier_ratio: 0.0,
        }
    }
}

/// Match two images' features and geometrically verify the result. Returns
/// `None` if too few putative matches exist, or too few survive RANSAC to
/// trust the pair (see `VerificationParams::min_inliers`/`min_inlier_ratio`) -
/// callers should simply not store the pair in that case, rather than storing
/// a low-confidence geometry.
pub fn match_and_verify(
    features1: &FeatureSet,
    features2: &FeatureSet,
    cam1: &CameraModel,
    cam2: &CameraModel,
    params: &VerificationParams,
) -> Option<TwoViewGeometryRecord> {
    let putative = match_descriptors(
        &features1.descriptors,
        &features2.descriptors,
        &params.match_params,
    );
    if putative.len() < params.min_inliers.max(8) {
        return None;
    }

    let pts1: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(i, _)| {
            let kp = features1.keypoints[i as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();
    let pts2: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(_, j)| {
            let kp = features2.keypoints[j as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();

    // ArUco marker corners are already unambiguous exact-ID correspondences
    // with (by construction) very few points per pair - RANSAC's minimal
    // sample of 8 is usually more points than a couple of markers provide.
    // Accept them directly (still gated by `min_inliers`) instead of running
    // essential-matrix RANSAC on a handful of points.
    if matches!(
        features1.descriptors,
        sfm_core::Descriptors::MarkerCorner { .. }
    ) {
        if putative.len() < params.min_inliers {
            return None;
        }
        return Some(TwoViewGeometryRecord {
            pose: sfm_core::Pose::identity(),
            inlier_matches: putative,
        });
    }

    let geometry = estimate_two_view_geometry(
        &pts1,
        &pts2,
        cam1,
        cam2,
        params.ransac_threshold_px,
        params.ransac_max_iterations,
    )?;

    let inlier_ratio = geometry.num_inliers as f64 / putative.len() as f64;
    if geometry.num_inliers < params.min_inliers || inlier_ratio < params.min_inlier_ratio {
        return None;
    }

    let inlier_matches: Vec<(u32, u32)> = putative
        .iter()
        .zip(geometry.inliers.iter())
        .filter_map(|(&m, &keep)| keep.then_some(m))
        .collect();

    Some(TwoViewGeometryRecord {
        pose: geometry.pose,
        inlier_matches,
    })
}

/// Convenience used by callers that already have pixel points in hand
/// (kept in sync with [`sfm_geometry::essential::to_normalized`] so
/// `sfm-reconstruction` doesn't need its own copy).
pub fn normalize_point(p: (f64, f64), cam: &CameraModel) -> (f64, f64) {
    to_normalized(p, cam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use sfm_core::{Keypoint, Pose};

    fn pinhole(f: f64, w: u32, h: u32) -> CameraModel {
        CameraModel::SimplePinhole {
            f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }

    /// A distinct, unambiguous descriptor per index (real SIFT/ORB descriptors
    /// are never exact duplicates across genuinely different keypoints, so
    /// matching them doesn't need to cope with that degenerate case here).
    fn descriptor_row(seed: usize, dim: usize) -> Vec<f32> {
        let mut row = vec![0f32; dim];
        row[seed] = 1.0;
        row
    }

    #[test]
    fn end_to_end_match_and_verify_on_synthetic_two_view() {
        let cam = pinhole(800.0, 640, 480);
        let true_translation = Vector3::new(1.0, 0.0, 0.0);
        let true_rotation = UnitQuaternion::from_euler_angles(0.0, 0.15, 0.0);
        let pose2_true = Pose::from_rotation_translation(true_rotation, true_translation);

        let points_3d: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.6 * (t * 0.9).sin(), 0.4 * (t * 1.3).cos(), 4.0 + 0.07 * t)
            })
            .collect();

        let mut kps1 = Vec::new();
        let mut kps2 = Vec::new();
        let mut desc1 = Vec::new();
        let mut desc2 = Vec::new();
        for (i, p) in points_3d.iter().enumerate() {
            let (x1, y1) = cam.project(p);
            let (x2, y2) = cam.project(&pose2_true.transform_point(p));
            kps1.push(Keypoint {
                x: x1 as f32,
                y: y1 as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            });
            kps2.push(Keypoint {
                x: x2 as f32,
                y: y2 as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            });
            desc1.extend_from_slice(&descriptor_row(i, points_3d.len()));
            desc2.extend_from_slice(&descriptor_row(i, points_3d.len()));
        }

        let dim = points_3d.len() as u32;
        let features1 = FeatureSet {
            keypoints: kps1,
            descriptors: sfm_core::Descriptors::Float32 { dim, data: desc1 },
        };
        let features2 = FeatureSet {
            keypoints: kps2,
            descriptors: sfm_core::Descriptors::Float32 { dim, data: desc2 },
        };

        let record = match_and_verify(
            &features1,
            &features2,
            &cam,
            &cam,
            &VerificationParams::default(),
        )
        .expect("should verify a clean synthetic two-view pair");
        assert!(
            record.inlier_matches.len() >= 35,
            "got {} inliers",
            record.inlier_matches.len()
        );

        let cos_angle = record
            .pose
            .translation
            .normalize()
            .dot(&true_translation.normalize())
            .abs();
        assert!(cos_angle > 0.99, "cos_angle={cos_angle}");
    }
}
