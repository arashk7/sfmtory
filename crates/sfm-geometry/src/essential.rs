//! Essential-matrix estimation and relative-pose recovery. Deliberately uses
//! the normalized 8-point solver (not a minimal 5-point solver like Nister's)
//! inside RANSAC: it needs more iterations to reach the same confidence at a
//! given inlier ratio, but Rust's speed leaves plenty of headroom for that,
//! and it lets `essential.rs` share `fundamental.rs`'s linear solver exactly
//! instead of a second, more intricate minimal solver. Accuracy is unaffected
//! since both this and a 5-point pipeline do a full inlier-set LO refit
//! afterward - only the RANSAC sample-efficiency differs. Revisit with a
//! 5-point minimal solver if profiling shows RANSAC iteration count actually
//! dominates `sfm match` runtime on hard (low-inlier-ratio) scenes.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use sfm_core::{CameraModel, Pose};

use crate::fundamental::{linear_eight_point, sampson_distance};
use crate::linalg::enforce_essential_constraint;
use crate::ransac::{ransac, RansacParams};
use crate::triangulation::triangulate_normalized;

/// Pixel -> normalized camera coordinates, ignoring distortion. Adequate for
/// RANSAC sampling and the initial pose estimate; `sfm-ba` refines the true
/// (possibly distorted) reprojection later. Distortion-aware undistortion
/// could replace this for cameras with large `k`, but isn't needed for the
/// pipeline to be correct end-to-end.
pub fn to_normalized(p: (f64, f64), cam: &CameraModel) -> (f64, f64) {
    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    ((p.0 - cx) / fx, (p.1 - cy) / fy)
}

pub fn estimate_essential(
    pts1_norm: &[(f64, f64)],
    pts2_norm: &[(f64, f64)],
) -> Option<Matrix3<f64>> {
    linear_eight_point(pts1_norm, pts2_norm).map(enforce_essential_constraint)
}

pub struct TwoViewGeometry {
    /// Relative pose of camera 2 with camera 1 fixed at the origin/identity.
    /// Translation is unit-scale (essential matrix has no scale information).
    pub pose: Pose,
    pub inliers: Vec<bool>,
    pub num_inliers: usize,
}

/// Full two-view pipeline: pixel correspondences + both cameras' intrinsics
/// in, RANSAC-verified relative pose out. `threshold_px` is an approximate
/// pixel reprojection tolerance, internally converted to the normalized-
/// coordinate Sampson threshold via the average focal length.
pub fn estimate_two_view_geometry(
    pts1_px: &[(f64, f64)],
    pts2_px: &[(f64, f64)],
    cam1: &CameraModel,
    cam2: &CameraModel,
    threshold_px: f64,
    max_iterations: usize,
) -> Option<TwoViewGeometry> {
    let n = pts1_px.len();
    if n < 8 || pts2_px.len() != n {
        return None;
    }
    let norm1: Vec<(f64, f64)> = pts1_px.iter().map(|&p| to_normalized(p, cam1)).collect();
    let norm2: Vec<(f64, f64)> = pts2_px.iter().map(|&p| to_normalized(p, cam2)).collect();

    let avg_focal = (cam1.focal_lengths().0 + cam2.focal_lengths().0) / 2.0;
    let threshold = threshold_px / avg_focal;
    let params = RansacParams {
        max_iterations,
        threshold,
        confidence: 0.999,
    };

    let (model, inliers) = ransac(
        n,
        8,
        &params,
        0xC0FFEE,
        |sample| {
            let s1: Vec<_> = sample.iter().map(|&i| norm1[i]).collect();
            let s2: Vec<_> = sample.iter().map(|&i| norm2[i]).collect();
            estimate_essential(&s1, &s2)
        },
        |e, i| sampson_distance(e, norm1[i], norm2[i]),
    )?;
    let original_count = inliers.iter().filter(|&&b| b).count();
    if original_count < 8 {
        return None;
    }

    // Local optimization: refit on the full inlier set (much better
    // conditioned than any single minimal 8-point sample) - but, as with
    // `pnp::pnp_ransac`'s identical structure, never trust the refit blindly:
    // a near-planar inlier set can make even this over-determined refit worse
    // than the minimal-sample model RANSAC already found. Fall back to the
    // original model+inliers if the refit doesn't improve on them.
    let inlier_idx: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
    let refit: Option<(Matrix3<f64>, Vec<bool>)> = (|| {
        let s1: Vec<_> = inlier_idx.iter().map(|&i| norm1[i]).collect();
        let s2: Vec<_> = inlier_idx.iter().map(|&i| norm2[i]).collect();
        let refined = estimate_essential(&s1, &s2)?;
        let final_inliers: Vec<bool> = (0..n)
            .map(|i| sampson_distance(&refined, norm1[i], norm2[i]) < threshold)
            .collect();
        Some((refined, final_inliers))
    })();
    let (refined, final_inliers) = match refit {
        Some((r, fi)) if fi.iter().filter(|&&b| b).count() >= original_count => (r, fi),
        _ => (model, inliers),
    };
    let final_idx: Vec<usize> = (0..n).filter(|&i| final_inliers[i]).collect();
    if final_idx.len() < 8 {
        return None;
    }

    let pose = recover_pose(&refined, &norm1, &norm2, &final_idx)?;
    let num_inliers = final_idx.len();
    Some(TwoViewGeometry {
        pose,
        inliers: final_inliers,
        num_inliers,
    })
}

/// Decompose `E` into the 4 candidate `(R, t)` relative poses and pick the
/// one under which the most inlier points triangulate in front of both
/// cameras (the standard cheirality check; Hartley & Zisserman ch. 9.6.3).
pub fn recover_pose(
    e: &Matrix3<f64>,
    norm1: &[(f64, f64)],
    norm2: &[(f64, f64)],
    inlier_idx: &[usize],
) -> Option<Pose> {
    let svd = e.svd(true, true);
    let mut u = svd.u?;
    let mut v_t = svd.v_t?;
    if u.determinant() < 0.0 {
        for i in 0..3 {
            u[(i, 2)] *= -1.0;
        }
    }
    if v_t.determinant() < 0.0 {
        for i in 0..3 {
            v_t[(2, i)] *= -1.0;
        }
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * v_t;
    let r2 = u * w.transpose() * v_t;
    let t: Vector3<f64> = u.column(2).into_owned();

    let candidates = [(r1, t), (r1, -t), (r2, t), (r2, -t)];
    let sample: Vec<usize> = inlier_idx.iter().copied().take(50).collect();

    let mut best_pose = None;
    let mut best_count = -1i32;
    for (r, t) in candidates.iter() {
        let pose2 = Pose::from_rotation_translation(
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r)),
            *t,
        );
        let mut count = 0;
        for &i in &sample {
            let views = [(Pose::identity(), norm1[i]), (pose2, norm2[i])];
            if let Some(p3d) = triangulate_normalized(&views) {
                let depth1 = p3d.z;
                let depth2 = pose2.transform_point(&p3d).z;
                if depth1 > 1e-6 && depth2 > 1e-6 {
                    count += 1;
                }
            }
        }
        if count > best_count {
            best_count = count;
            best_pose = Some(pose2);
        }
    }
    best_pose
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfm_core::CameraModel;

    fn pinhole(f: f64, w: u32, h: u32) -> CameraModel {
        CameraModel::SimplePinhole {
            f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }

    #[test]
    fn recovers_relative_pose_from_synthetic_sideways_translation() {
        let cam = pinhole(800.0, 640, 480);
        let true_translation = Vector3::new(1.0, 0.0, 0.0);
        let true_rotation = UnitQuaternion::from_euler_angles(0.0, 0.1, 0.0);
        let pose2_true = Pose::from_rotation_translation(true_rotation, true_translation);

        let points_3d: Vec<Vector3<f64>> = (0..60)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.6 * (t * 0.9).sin(), 0.4 * (t * 1.7).cos(), 4.0 + 0.05 * t)
            })
            .collect();

        let project = |p: &Vector3<f64>, cam: &CameraModel| -> (f64, f64) { cam.project(p) };

        let pts1: Vec<(f64, f64)> = points_3d.iter().map(|p| project(p, &cam)).collect();
        let pts2: Vec<(f64, f64)> = points_3d
            .iter()
            .map(|p| project(&pose2_true.transform_point(p), &cam))
            .collect();

        let geom = estimate_two_view_geometry(&pts1, &pts2, &cam, &cam, 1.0, 2000)
            .expect("should recover two-view geometry");
        assert!(
            geom.num_inliers >= 50,
            "expected most points to be inliers, got {}",
            geom.num_inliers
        );

        // Translation direction should match up to the essential matrix's
        // inherent scale ambiguity (both sides unit-normalized).
        let recovered_dir = geom.pose.translation.normalize();
        let true_dir = true_translation.normalize();
        let cos_angle = recovered_dir.dot(&true_dir).abs();
        assert!(
            cos_angle > 0.99,
            "translation direction off: cos={cos_angle}"
        );

        let angle_error = geom.pose.rotation.angle_to(&true_rotation);
        assert!(
            angle_error < 0.02,
            "rotation error too high: {angle_error} rad"
        );
    }
}
