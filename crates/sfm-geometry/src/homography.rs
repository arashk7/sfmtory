//! Planar two-view geometry: homography estimation and decomposition into a
//! relative pose.
//!
//! The essential matrix is *degenerate* when the observed points are coplanar.
//! Every point on a plane satisfies infinitely many epipolar geometries, so an
//! eight-point fit has nothing to pin it down: RANSAC settles on whichever
//! spurious solution its sample suggested, discards most of the genuinely
//! correct correspondences as "outliers", and returns a relative pose that is
//! simply wrong. This is not a robustness problem to be tuned around - it is
//! the model being unidentifiable on that data.
//!
//! Planar scenes are common and often deliberate: a printed calibration board,
//! a fiducial grid, a facade. For those the right model is a homography, which
//! *is* identifiable, and which decomposes into the same `(R, t)` the caller
//! wanted from the essential matrix. `sfm-match` estimates both and prefers
//! this path when the homography explains the data at least as well (see
//! `estimate_two_view_geometry`).

use nalgebra::{Matrix3, Vector3};
use sfm_core::Pose;

use crate::ransac::{ransac, RansacParams};

/// Direct linear transform for `H` mapping `pts1 -> pts2`, from >= 4
/// correspondences in normalized camera coordinates.
pub fn linear_homography(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> Option<Matrix3<f64>> {
    let n = pts1.len();
    if n < 4 || pts2.len() != n {
        return None;
    }
    // At least 9 rows: with the minimal four correspondences the constraint
    // matrix is 8x9, and a thin SVD of that yields only 8 right-singular
    // vectors - there would be no ninth row to read the null space from.
    // Padding with zeros leaves the solution unchanged.
    let mut a = nalgebra::DMatrix::<f64>::zeros((2 * n).max(9), 9);
    for i in 0..n {
        let (x, y) = pts1[i];
        let (u, v) = pts2[i];
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let h = vt.row(8);
    let m = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(m)
}

/// Symmetric transfer error of one correspondence under `H`, in normalized
/// units. Symmetric rather than one-way so a homography that happens to be
/// well conditioned in one direction only cannot score well.
pub fn homography_transfer_error(h: &Matrix3<f64>, p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let fwd = h * Vector3::new(p1.0, p1.1, 1.0);
    let Some(hinv) = h.try_inverse() else {
        return f64::MAX;
    };
    let bwd = hinv * Vector3::new(p2.0, p2.1, 1.0);
    if fwd.z.abs() < 1e-12 || bwd.z.abs() < 1e-12 {
        return f64::MAX;
    }
    let e1 = ((fwd.x / fwd.z - p2.0).powi(2) + (fwd.y / fwd.z - p2.1).powi(2)).sqrt();
    let e2 = ((bwd.x / bwd.z - p1.0).powi(2) + (bwd.y / bwd.z - p1.1).powi(2)).sqrt();
    0.5 * (e1 + e2)
}

/// RANSAC homography over normalized correspondences.
pub fn estimate_homography_ransac(
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    threshold: f64,
    max_iterations: usize,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    let n = pts1.len();
    if n < 4 {
        return None;
    }
    let params = RansacParams {
        max_iterations,
        threshold,
        confidence: 0.999,
    };
    let (model, inliers) = ransac(
        n,
        4,
        &params,
        0xB0A7,
        |sample| {
            let s1: Vec<_> = sample.iter().map(|&i| pts1[i]).collect();
            let s2: Vec<_> = sample.iter().map(|&i| pts2[i]).collect();
            linear_homography(&s1, &s2)
        },
        |h, i| homography_transfer_error(h, pts1[i], pts2[i]),
    )?;
    // Refit on the full inlier set, keeping the refit only if it does not lose
    // inliers - same guard as the essential-matrix path uses.
    let idx: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
    let before = idx.len();
    if before < 4 {
        return None;
    }
    let s1: Vec<_> = idx.iter().map(|&i| pts1[i]).collect();
    let s2: Vec<_> = idx.iter().map(|&i| pts2[i]).collect();
    if let Some(refined) = linear_homography(&s1, &s2) {
        let refit: Vec<bool> = (0..n)
            .map(|i| homography_transfer_error(&refined, pts1[i], pts2[i]) < threshold)
            .collect();
        if refit.iter().filter(|&&b| b).count() >= before {
            return Some((refined, refit));
        }
    }
    Some((model, inliers))
}

/// Decomposes a normalized-coordinate homography into the relative pose that
/// produced it, choosing among the algebraic solutions by cheirality.
///
/// Uses the classical Faugeras/Lustman SVD construction: a homography induced
/// by a plane admits up to eight `(R, t, n)` solutions, of which four survive
/// the requirement that the plane lie in front of the first camera, and the
/// ambiguity is resolved by counting how many correspondences triangulate in
/// front of *both* cameras. Translation is recovered only up to scale, exactly
/// as with the essential matrix, so the caller's usual unit-baseline
/// convention still holds.
pub fn decompose_homography(
    h: &Matrix3<f64>,
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    inliers: &[bool],
) -> Option<Pose> {
    // Scale so the middle singular value is 1; the decomposition below is
    // written for that normalization.
    let svd = h.svd(true, true);
    let d = svd.singular_values;
    if d[1].abs() < 1e-12 {
        return None;
    }
    let hn = h / d[1];

    let svd = hn.svd(true, true);
    let mut u = svd.u?;
    let mut vt = svd.v_t?;
    let d = svd.singular_values;
    let (d1, d2, d3) = (d[0], d[1], d[2]);
    // Force right-handed factors so the recovered R is a rotation, not a
    // reflection.
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let v = vt.transpose();

    if (d1 - d3).abs() < 1e-9 {
        // Pure rotation (or a plane at infinity): no translation is
        // recoverable, and such a pair cannot triangulate anything anyway.
        return None;
    }

    let aux1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).max(0.0).sqrt();
    let aux3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).max(0.0).sqrt();
    let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];

    let mut candidates: Vec<Pose> = Vec::with_capacity(8);

    // d' = +d2 branch.
    let aux_st = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).max(0.0).sqrt() / ((d1 + d3) * d2);
    let ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    for (s1, s3) in signs {
        let stheta = s1 * s3 * aux_st;
        let rp = Matrix3::new(ctheta, 0.0, -stheta, 0.0, 1.0, 0.0, stheta, 0.0, ctheta);
        let r = u * rp * vt;
        let tp = Vector3::new(s1 * aux1, 0.0, -s3 * aux3) * (d1 - d3);
        let t = (u * tp).normalize();
        if t.iter().all(|c| c.is_finite()) {
            candidates.push(pose_from(r, t));
        }
    }

    // d' = -d2 branch.
    let aux_sp = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).max(0.0).sqrt() / ((d1 - d3) * d2);
    let cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    for (s1, s3) in signs {
        let sphi = s1 * s3 * aux_sp;
        let rp = Matrix3::new(cphi, 0.0, sphi, 0.0, -1.0, 0.0, sphi, 0.0, -cphi);
        let r = u * rp * vt;
        let tp = Vector3::new(s1 * aux1, 0.0, s3 * aux3) * (d1 + d3);
        let t = (u * tp).normalize();
        if t.iter().all(|c| c.is_finite()) {
            candidates.push(pose_from(r, t));
        }
    }
    let _ = v;

    // Cheirality vote over the inlier correspondences.
    let mut best: Option<(usize, Pose)> = None;
    for cand in candidates {
        let mut good = 0usize;
        for i in 0..pts1.len() {
            if !inliers.get(i).copied().unwrap_or(false) {
                continue;
            }
            let Some(xyz) = crate::triangulation::triangulate_normalized(&[
                (Pose::identity(), pts1[i]),
                (cand, pts2[i]),
            ]) else {
                continue;
            };
            if xyz.z > 0.0 && cand.transform_point(&xyz).z > 0.0 {
                good += 1;
            }
        }
        if best.as_ref().map(|(b, _)| good > *b).unwrap_or(true) {
            best = Some((good, cand));
        }
    }
    let (good, pose) = best?;
    if good == 0 {
        return None;
    }
    Some(pose)
}

fn pose_from(r: Matrix3<f64>, t: Vector3<f64>) -> Pose {
    let rot = nalgebra::Rotation3::from_matrix_unchecked(r);
    Pose::from_rotation_translation(nalgebra::UnitQuaternion::from_rotation_matrix(&rot), t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic planar scene viewed from two poses - the configuration that
    /// defeats the essential matrix.
    fn planar_pair() -> (Vec<(f64, f64)>, Vec<(f64, f64)>, Pose) {
        let pose2 = Pose::from_rotation_translation(
            nalgebra::UnitQuaternion::from_euler_angles(0.05, 0.35, -0.02),
            Vector3::new(-0.8, 0.05, 0.1),
        );
        let mut p1 = Vec::new();
        let mut p2 = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                // All points on the plane z = 4.
                let x = -1.0 + 0.4 * i as f64;
                let y = -1.0 + 0.4 * j as f64;
                let w = Vector3::new(x, y, 4.0);
                p1.push((w.x / w.z, w.y / w.z));
                let c = pose2.transform_point(&w);
                p2.push((c.x / c.z, c.y / c.z));
            }
        }
        (p1, p2, pose2)
    }

    #[test]
    fn homography_fits_a_planar_pair_exactly() {
        let (p1, p2, _) = planar_pair();
        let h = linear_homography(&p1, &p2).unwrap();
        for i in 0..p1.len() {
            assert!(homography_transfer_error(&h, p1[i], p2[i]) < 1e-8);
        }
    }

    #[test]
    fn ransac_finds_the_homography_among_outliers() {
        let (mut p1, mut p2, _) = planar_pair();
        for k in 0..8 {
            p1.push((0.1 * k as f64, -0.2));
            p2.push((0.7 - 0.05 * k as f64, 0.9));
        }
        let (h, inl) = estimate_homography_ransac(&p1, &p2, 1e-3, 2000).unwrap();
        let n_in = inl.iter().filter(|&&b| b).count();
        assert!(n_in >= 36, "expected the 36 planar points as inliers, got {n_in}");
        assert!(homography_transfer_error(&h, p1[0], p2[0]) < 1e-3);
    }

    #[test]
    fn decomposition_recovers_the_relative_pose() {
        let (p1, p2, truth) = planar_pair();
        let h = linear_homography(&p1, &p2).unwrap();
        let inliers = vec![true; p1.len()];
        let pose = decompose_homography(&h, &p1, &p2, &inliers).unwrap();

        // Rotation should match closely.
        let dr = pose.rotation.rotation_to(&truth.rotation).angle().to_degrees();
        assert!(dr < 1.0, "rotation off by {dr} deg");
        // Translation is only up to scale, so compare directions.
        let dot = pose.translation.normalize().dot(&truth.translation.normalize());
        assert!(dot > 0.99, "translation direction off, cos = {dot}");
    }
}
