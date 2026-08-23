//! Linear (normalized) 8-point solver, shared by fundamental-matrix
//! estimation (`sfm-match`'s pixel-space geometric verification) and
//! essential-matrix estimation (`essential.rs`, on calibrated coordinates) -
//! the two only differ in which constraint gets enforced on the result, not
//! in how the linear system is built or solved.

use nalgebra::{DMatrix, Matrix3};

use crate::linalg::{enforce_rank2, null_space_vector};

/// Hartley normalization: translate to centroid, scale so the average point
/// distance from the origin is `sqrt(2)`. Critical for the 8-point
/// algorithm's numerical conditioning - without it, the linear system is
/// badly scaled (pixel coordinates range over hundreds/thousands) and the
/// SVD null-space solve becomes noise-dominated.
pub fn hartley_normalize(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (sx, sy) = pts
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (mx, my) = (sx / n, sy / n);
    let mean_dist = pts
        .iter()
        .map(|(x, y)| ((x - mx).powi(2) + (y - my).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let scale = if mean_dist > 1e-12 {
        2f64.sqrt() / mean_dist
    } else {
        1.0
    };
    let t = Matrix3::new(
        scale,
        0.0,
        -scale * mx,
        0.0,
        scale,
        -scale * my,
        0.0,
        0.0,
        1.0,
    );
    let normalized = pts
        .iter()
        .map(|(x, y)| (scale * (x - mx), scale * (y - my)))
        .collect();
    (normalized, t)
}

/// The raw (unconstrained, generically rank-3) linear 8-point solution in
/// the *original* coordinate space of `pts1`/`pts2` - Hartley normalization
/// is applied internally for conditioning and undone before returning.
/// Callers enforce whichever manifold constraint applies to their matrix
/// (see module docs).
pub fn linear_eight_point(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> Option<Matrix3<f64>> {
    let n = pts1.len();
    if n < 8 || pts2.len() != n {
        return None;
    }
    let (norm1, t1) = hartley_normalize(pts1);
    let (norm2, t2) = hartley_normalize(pts2);

    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let (x1, y1) = norm1[i];
        let (x2, y2) = norm2[i];
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }
    let f_vec = null_space_vector(a)?;
    let f_norm = Matrix3::new(
        f_vec[0], f_vec[1], f_vec[2], f_vec[3], f_vec[4], f_vec[5], f_vec[6], f_vec[7], f_vec[8],
    );
    Some(t2.transpose() * f_norm * t1)
}

/// Fundamental matrix from pixel-space correspondences: linear solve + rank-2
/// projection. Not used for pose recovery directly (see `essential.rs`) but
/// useful as a calibration-free geometric-verification check.
pub fn estimate_fundamental(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> Option<Matrix3<f64>> {
    linear_eight_point(pts1, pts2).map(enforce_rank2)
}

/// Sampson distance: a first-order approximation of the geometric reprojection
/// error implied by the epipolar constraint `x2^T F x1 = 0`. Cheap to compute
/// (no triangulation needed) so it's the standard RANSAC scoring residual for
/// both F and E.
pub fn sampson_distance(f: &Matrix3<f64>, p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let x1 = nalgebra::Vector3::new(p1.0, p1.1, 1.0);
    let x2 = nalgebra::Vector3::new(p2.0, p2.1, 1.0);
    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;
    let x2tfx1 = x2.dot(&fx1);
    let denom = fx1.x.powi(2) + fx1.y.powi(2) + ftx2.x.powi(2) + ftx2.y.powi(2);
    if denom < 1e-12 {
        return f64::MAX;
    }
    (x2tfx1 * x2tfx1 / denom).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_fundamental_matrix_from_synthetic_pure_translation() {
        // Camera1 at origin looking down +Z, camera2 translated along +X by 1.
        // Under this simple rig the epipolar lines are horizontal, so we can
        // sanity-check the epipolar constraint directly rather than needing a
        // full projective setup.
        let points_3d: Vec<(f64, f64, f64)> = (0..20)
            .map(|i| {
                let t = i as f64;
                (0.3 * (t * 0.7).sin(), 0.3 * (t * 1.3).cos(), 3.0 + 0.1 * t)
            })
            .collect();
        let f = 800.0;
        let (cx, cy) = (320.0, 240.0);
        let project = |(x, y, z): (f64, f64, f64), tx: f64| ((x - tx) / z * f + cx, y / z * f + cy);

        let pts1: Vec<(f64, f64)> = points_3d.iter().map(|&p| project(p, 0.0)).collect();
        let pts2: Vec<(f64, f64)> = points_3d.iter().map(|&p| project(p, -1.0)).collect();

        let f_mat = estimate_fundamental(&pts1, &pts2).expect("should estimate F");
        for i in 0..pts1.len() {
            let residual = sampson_distance(&f_mat, pts1[i], pts2[i]);
            assert!(residual < 1e-3, "residual too high at {i}: {residual}");
        }
    }
}
