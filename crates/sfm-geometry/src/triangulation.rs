//! Multi-view DLT triangulation, operating in normalized (calibrated) camera
//! coordinates so it's agnostic to the specific camera model - callers
//! convert pixel observations via [`crate::essential::to_normalized`] first.

use nalgebra::{DMatrix, Vector3};
use sfm_core::Pose;

use crate::linalg::null_space_vector;

/// Triangulate a single 3D point from 2+ views via linear DLT (smallest
/// right-singular-vector solve of the stacked cross-product constraints).
/// Returns `None` for degenerate configurations (fewer than 2 views, or a
/// numerically singular system).
pub fn triangulate_normalized(views: &[(Pose, (f64, f64))]) -> Option<Vector3<f64>> {
    if views.len() < 2 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(views.len() * 2, 4);
    for (i, (pose, (x, y))) in views.iter().enumerate() {
        let p = pose.to_matrix();
        for c in 0..4 {
            let row0 = p[(0, c)];
            let row1 = p[(1, c)];
            let row2 = p[(2, c)];
            a[(2 * i, c)] = x * row2 - row0;
            a[(2 * i + 1, c)] = y * row2 - row1;
        }
    }
    let x_h = null_space_vector(a)?;
    if x_h[3].abs() < 1e-12 {
        return None;
    }
    Some(Vector3::new(
        x_h[0] / x_h[3],
        x_h[1] / x_h[3],
        x_h[2] / x_h[3],
    ))
}

/// Reprojection error in normalized-coordinate units (radians-ish for small
/// angles; multiply by a camera's focal length to convert to pixels).
pub fn reprojection_error_normalized(
    pose: &Pose,
    point: &Vector3<f64>,
    observed: (f64, f64),
) -> Option<f64> {
    let pc = pose.transform_point(point);
    if pc.z <= 1e-9 {
        return None;
    }
    let dx = pc.x / pc.z - observed.0;
    let dy = pc.y / pc.z - observed.1;
    Some((dx * dx + dy * dy).sqrt())
}

/// Triangulation angle (radians) between two viewing rays to the same point -
/// low angle means the point is poorly constrained (near-parallel rays, as
/// happens for distant points or a very short camera baseline) and should be
/// filtered out before bundle adjustment.
pub fn triangulation_angle(
    camera_center_1: &Vector3<f64>,
    camera_center_2: &Vector3<f64>,
    point: &Vector3<f64>,
) -> f64 {
    let d1 = (point - camera_center_1).normalize();
    let d2 = (point - camera_center_2).normalize();
    d1.dot(&d2).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulates_known_point_from_two_views() {
        let true_point = Vector3::new(0.3, -0.2, 5.0);
        let pose1 = Pose::identity();
        let pose2 = Pose::from_rotation_translation(
            nalgebra::UnitQuaternion::identity(),
            Vector3::new(1.0, 0.0, 0.0),
        );

        let obs1 = {
            let p = pose1.transform_point(&true_point);
            (p.x / p.z, p.y / p.z)
        };
        let obs2 = {
            let p = pose2.transform_point(&true_point);
            (p.x / p.z, p.y / p.z)
        };

        let recovered = triangulate_normalized(&[(pose1, obs1), (pose2, obs2)]).unwrap();
        assert!(
            (recovered - true_point).norm() < 1e-9,
            "recovered={recovered:?}"
        );
    }

    #[test]
    fn triangulation_angle_is_zero_for_colinear_setup() {
        let point = Vector3::new(0.0, 0.0, 10.0);
        let c1 = Vector3::new(0.0, 0.0, 0.0);
        let c2 = Vector3::new(0.0, 0.0, 5.0); // same ray toward the point
        let angle = triangulation_angle(&c1, &c2, &point);
        assert!(angle < 1e-9, "angle={angle}");
    }
}
