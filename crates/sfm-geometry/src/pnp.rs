//! Linear (DLT) camera resectioning: given 3D-2D correspondences in
//! normalized camera coordinates, solve for the camera's world-to-camera
//! pose. Used by the incremental pipeline to register a new image against
//! points already triangulated from previously-registered images.
//!
//! Deliberately the linear 6-point DLT rather than a minimal 3-point P3P
//! solver, for the same reason `essential.rs` uses the linear 8-point over a
//! minimal 5-point solver: simpler to implement correctly, and RANSAC just
//! needs a few more iterations at typical (already-verified, high-inlier-
//! ratio) registration-stage correspondence sets to compensate.

use nalgebra::{DMatrix, Matrix3, SMatrix, Vector3};
use sfm_core::Pose;

use crate::linalg::null_space_vector;
use crate::ransac::{ransac, RansacParams};

/// Solve for a single camera pose via linear DLT from `points3d[i] <->
/// points2d_norm[i]` correspondences (needs at least 6, since we solve for
/// all 12 entries of the 3x4 projection matrix rather than exploiting the
/// rotation's orthogonality constraint the way a minimal solver would).
pub fn pnp_dlt(points3d: &[Vector3<f64>], points2d_norm: &[(f64, f64)]) -> Option<Pose> {
    let n = points3d.len();
    if n < 6 || points2d_norm.len() != n {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let x = points3d[i];
        let (u, v) = points2d_norm[i];
        a[(2 * i, 0)] = x.x;
        a[(2 * i, 1)] = x.y;
        a[(2 * i, 2)] = x.z;
        a[(2 * i, 3)] = 1.0;
        a[(2 * i, 8)] = -u * x.x;
        a[(2 * i, 9)] = -u * x.y;
        a[(2 * i, 10)] = -u * x.z;
        a[(2 * i, 11)] = -u;

        a[(2 * i + 1, 4)] = x.x;
        a[(2 * i + 1, 5)] = x.y;
        a[(2 * i + 1, 6)] = x.z;
        a[(2 * i + 1, 7)] = 1.0;
        a[(2 * i + 1, 8)] = -v * x.x;
        a[(2 * i + 1, 9)] = -v * x.y;
        a[(2 * i + 1, 10)] = -v * x.z;
        a[(2 * i + 1, 11)] = -v;
    }
    let p_vec = null_space_vector(a)?;
    let mut p = SMatrix::<f64, 3, 4>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            p[(r, c)] = p_vec[r * 4 + c];
        }
    }

    let r_block: Matrix3<f64> = p.fixed_view::<3, 3>(0, 0).into_owned();
    let svd = r_block.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    // `r_block` is (up to noise) an exact scalar multiple of a rotation:
    // `r_block = lambda * R_true`. Its singular values are then all equal
    // (`|lambda|`), which makes the SVD's own `U`/`V` individually
    // non-unique - but `U * V^T` is still exactly `sign(lambda) * R_true`
    // regardless of which valid U/V the solver returns, since Sigma is a
    // multiple of the identity and so commutes with any change of basis.
    // That means the textbook Procrustes "closest rotation" fix (flip the
    // smallest singular value's sign) is the wrong tool here - it's designed
    // for `r_block` being merely *close to* a rotation (e.g. Kabsch
    // alignment), not an exact multiple of one. The correct fix for our
    // exact-scalar case is simpler: flip the whole candidate's sign.
    let candidate = u * v_t;
    let r = if candidate.determinant() < 0.0 {
        -candidate
    } else {
        candidate
    };

    // r_block ~= lambda * r for some scalar lambda (the DLT null vector's
    // homogeneous scale, sign included); recover it by least-squares
    // projection rather than guessing, since `r` alone can't tell us the
    // sign or magnitude that r_block actually carries.
    let numerator: f64 = r_block.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
    let denominator: f64 = r.iter().map(|v| v * v).sum();
    if denominator < 1e-12 {
        return None;
    }
    let lambda = numerator / denominator;
    if lambda.abs() < 1e-12 {
        return None;
    }
    let t = Vector3::new(p[(0, 3)], p[(1, 3)], p[(2, 3)]) / lambda;

    Some(Pose::from_rotation_translation(
        nalgebra::UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(r),
        ),
        t,
    ))
}

/// Rejects minimal samples whose 3D points are (near-)coplanar: linear
/// PnP-DLT solves for a full 11-DOF projective camera matrix, but coplanar
/// 3D points only constrain an 8-DOF homography, so the 12-unknown linear
/// system genuinely loses rank rather than merely becoming ill-conditioned
/// (Hartley & Zisserman's "critical configurations" for resectioning). A
/// facade/wall-dominated scene produces exactly this: most 6-point samples
/// drawn from points on the wall are coplanar, and `pnp_dlt` silently
/// returns *some* pose from a rank-deficient system rather than failing -
/// often one only a handful of points happen to agree with by chance,
/// which was observed corrupting the model (e.g. a "registered" image with
/// 9 inliers out of 212 correspondences). Scored via the ratio of the
/// smallest to largest eigenvalue of the sample's covariance: a genuinely
/// flat sample has near-zero out-of-plane variance relative to its in-plane
/// spread, regardless of the sample's absolute scale.
fn is_well_conditioned_for_dlt(points3d: &[Vector3<f64>]) -> bool {
    let n = points3d.len();
    let centroid = points3d.iter().sum::<Vector3<f64>>() / n as f64;
    let mut cov = Matrix3::<f64>::zeros();
    for p in points3d {
        let d = p - centroid;
        cov += d * d.transpose();
    }
    let eigenvalues = cov.symmetric_eigenvalues();
    let largest = eigenvalues.max();
    let smallest = eigenvalues.min();
    if largest < 1e-12 {
        return false;
    }
    smallest / largest > 1e-3
}

fn reprojection_residual(pose: &Pose, point: &Vector3<f64>, observed: (f64, f64)) -> f64 {
    let pc = pose.transform_point(point);
    if pc.z <= 1e-9 {
        return f64::MAX;
    }
    let dx = pc.x / pc.z - observed.0;
    let dy = pc.y / pc.z - observed.1;
    (dx * dx + dy * dy).sqrt()
}

/// Gauss-Newton reprojection-error refinement of a pose, given a fixed set
/// of 3D points and their normalized 2D observations. Unlike `pnp_dlt`
/// (which minimizes an algebraic error and is only exact for noise-free
/// data), this minimizes actual reprojection error directly, and typically
/// pulls a pose noticeably closer to the true optimum when the DLT estimate
/// was built from noisy triangulated points - which in turn recovers
/// inliers the linear estimate's larger residuals had pushed just past the
/// RANSAC threshold. Left-multiplicative SO(3) perturbation for rotation
/// (`R_new = exp(delta) * R`), plain additive update for translation.
pub fn refine_pose_gauss_newton(
    initial: &Pose,
    points3d: &[Vector3<f64>],
    points2d_norm: &[(f64, f64)],
    iterations: usize,
) -> Pose {
    let mut pose = *initial;
    let n = points3d.len();
    for _ in 0..iterations {
        let mut jtj = nalgebra::Matrix6::<f64>::zeros();
        let mut jtr = nalgebra::Vector6::<f64>::zeros();
        for i in 0..n {
            let pc = pose.transform_point(&points3d[i]);
            if pc.z <= 1e-9 {
                continue;
            }
            let inv_z = 1.0 / pc.z;
            let x = pc.x * inv_z;
            let y = pc.y * inv_z;
            let (ou, ov) = points2d_norm[i];

            // d(pc)/d(delta_rot) = -skew(pc) for a left-multiplicative
            // perturbation; d(pc)/d(delta_t) = I3.
            let d_pc_d_rot = Matrix3::new(0.0, pc.z, -pc.y, -pc.z, 0.0, pc.x, pc.y, -pc.x, 0.0);

            let mut j = SMatrix::<f64, 2, 6>::zeros();
            for c in 0..3 {
                let dpc = d_pc_d_rot.column(c);
                j[(0, c)] = inv_z * dpc.x - x * inv_z * dpc.z;
                j[(1, c)] = inv_z * dpc.y - y * inv_z * dpc.z;
            }
            j[(0, 3)] = inv_z;
            j[(1, 4)] = inv_z;
            j[(0, 5)] = -x * inv_z;
            j[(1, 5)] = -y * inv_z;

            let r = nalgebra::Vector2::new(x - ou, y - ov);
            jtj += j.transpose() * j;
            jtr += j.transpose() * r;
        }
        for d in 0..6 {
            jtj[(d, d)] += 1e-9;
        }
        let Some(delta) = jtj.lu().solve(&(-jtr)) else {
            break;
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }
        let delta_rot = Vector3::new(delta[0], delta[1], delta[2]);
        let delta_t = Vector3::new(delta[3], delta[4], delta[5]);
        pose.rotation = nalgebra::UnitQuaternion::from_scaled_axis(delta_rot) * pose.rotation;
        pose.translation += delta_t;
    }
    pose
}

/// RANSAC-wrapped PnP with an inlier-set LO refit, mirroring
/// `essential::estimate_two_view_geometry`'s structure. `threshold` is in
/// normalized-coordinate units (see `essential::to_normalized`).
/// Fits a plane to `points3d` and returns an orthonormal frame
/// `(centroid, e1, e2, e3)` with `e3` the normal, plus the plane's relative
/// thickness (smallest extent over largest). A small thickness means the
/// points really are coplanar.
fn plane_frame(points3d: &[Vector3<f64>]) -> Option<(Vector3<f64>, Matrix3<f64>, f64)> {
    let n = points3d.len();
    if n < 4 {
        return None;
    }
    let centroid = points3d.iter().sum::<Vector3<f64>>() / n as f64;
    let mut cov = Matrix3::zeros();
    for p in points3d {
        let d = p - centroid;
        cov += d * d.transpose();
    }
    let eig = cov.symmetric_eigen();
    // Ascending order: the smallest eigenvalue's eigenvector is the normal.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let (i_min, i_mid, i_max) = (idx[0], idx[1], idx[2]);
    let largest = eig.eigenvalues[i_max];
    if largest <= 1e-18 {
        return None;
    }
    let thickness = (eig.eigenvalues[i_min] / largest).max(0.0).sqrt();
    let e1 = eig.eigenvectors.column(i_max).normalize();
    let e2 = eig.eigenvectors.column(i_mid).normalize();
    let e3 = e1.cross(&e2).normalize();
    let basis = Matrix3::from_columns(&[e1, e2, e3]);
    Some((centroid, basis, thickness))
}

/// Pose from *coplanar* 3D points, via the plane-to-image homography.
///
/// Linear PnP-DLT solves for a full 11-DOF projective camera and is rank
/// deficient when the 3D points share a plane - which is exactly the case for
/// the most common calibration setup there is (a printed board, a fiducial
/// grid, a marker sheet on a screen). The pipeline already refused such
/// samples rather than returning a wrong pose, which is correct but leaves a
/// fully planar scene with no usable PnP at all: every image then falls back
/// to the weaker bootstrap path, and observed on real fiducial photos that
/// left every track at length two.
///
/// For a plane the right model is a homography, and it is exactly
/// determined: writing points in a 2D frame on the plane, the map to
/// normalized image coordinates is `H ~ [R e1 | R e2 | R c + t]`, from which
/// the rotation and translation follow directly. This is the same
/// construction Zhang's calibration method uses.
pub fn pnp_planar(points3d: &[Vector3<f64>], points2d_norm: &[(f64, f64)]) -> Option<Pose> {
    let n = points3d.len();
    if n < 4 || points2d_norm.len() != n {
        return None;
    }
    let (centroid, basis, _) = plane_frame(points3d)?;
    let (e1, e2) = (basis.column(0).into_owned(), basis.column(1).into_owned());

    // Plane-local 2D coordinates.
    let local: Vec<(f64, f64)> = points3d
        .iter()
        .map(|p| {
            let d = p - centroid;
            (d.dot(&e1), d.dot(&e2))
        })
        .collect();

    let h = crate::homography::linear_homography(&local, points2d_norm)?;
    let (h1, h2, h3) = (
        h.column(0).into_owned(),
        h.column(1).into_owned(),
        h.column(2).into_owned(),
    );
    // `H` is known only up to scale; both first columns are unit rotation
    // columns, so their average norm recovers it.
    let scale = 2.0 / (h1.norm() + h2.norm());
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    // Sign: the plane must sit in front of the camera.
    let scale = if (h3 * scale).z < 0.0 { -scale } else { scale };

    let r1 = h1 * scale;
    let r2 = h2 * scale;
    let t_local = h3 * scale;
    let r3 = r1.cross(&r2);
    // Nearest true rotation to the (noisy, not exactly orthonormal) columns.
    let approx = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = approx.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut r_plane = u * v_t;
    if r_plane.determinant() < 0.0 {
        let mut uu = u;
        uu.column_mut(2).neg_mut();
        r_plane = uu * v_t;
    }

    // `r_plane` maps the plane frame into the camera; compose with the plane
    // frame's own orientation to get the world-to-camera rotation, then undo
    // the centroid offset the local coordinates were measured from.
    let r_world = r_plane * basis.transpose();
    let t_world = t_local - r_world * centroid;
    if !r_world.iter().all(|v| v.is_finite()) || !t_world.iter().all(|v| v.is_finite()) {
        return None;
    }
    let rot = nalgebra::Rotation3::from_matrix_unchecked(r_world);
    Some(Pose::from_rotation_translation(
        nalgebra::UnitQuaternion::from_rotation_matrix(&rot),
        t_world,
    ))
}

/// Relative plane thickness below which a point set is treated as coplanar and
/// solved with `pnp_planar` instead of DLT.
const PLANAR_THICKNESS: f64 = 0.02;

pub fn pnp_ransac(
    points3d: &[Vector3<f64>],
    points2d_norm: &[(f64, f64)],
    threshold: f64,
    max_iterations: usize,
) -> Option<(Pose, Vec<bool>)> {
    let n = points3d.len();
    if n < 4 || points2d_norm.len() != n {
        return None;
    }
    let params = RansacParams {
        max_iterations,
        threshold,
        confidence: 0.999,
    };

    // Planar scenes get the homography-based solver: DLT is rank deficient
    // there, and refusing every sample (which is what the conditioning check
    // below does, correctly) would otherwise leave a planar target with no
    // PnP at all. Four points suffice for a homography, against DLT's six.
    let planar = plane_frame(points3d)
        .map(|(_, _, thickness)| thickness < PLANAR_THICKNESS)
        .unwrap_or(false);
    if planar {
        let (model, inliers) = ransac(
            n,
            4,
            &params,
            0xFACADE,
            |sample| {
                let p3: Vec<_> = sample.iter().map(|&i| points3d[i]).collect();
                let p2: Vec<_> = sample.iter().map(|&i| points2d_norm[i]).collect();
                pnp_planar(&p3, &p2)
            },
            |pose, i| reprojection_residual(pose, &points3d[i], points2d_norm[i]),
        )?;
        let count = inliers.iter().filter(|&&b| b).count();
        if count < 4 {
            return None;
        }
        // Refit on all inliers, then polish - same shape as the DLT path, and
        // accepted only if it doesn't lose inliers.
        let idx: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
        let p3: Vec<Vector3<f64>> = idx.iter().map(|&i| points3d[i]).collect();
        let p2: Vec<(f64, f64)> = idx.iter().map(|&i| points2d_norm[i]).collect();
        let (mut pose, mut best) = (model, inliers);
        if let Some(refined) = pnp_planar(&p3, &p2) {
            let refit: Vec<bool> = (0..n)
                .map(|i| reprojection_residual(&refined, &points3d[i], points2d_norm[i]) < threshold)
                .collect();
            if refit.iter().filter(|&&b| b).count() >= count {
                pose = refined;
                best = refit;
            }
        }
        let idx: Vec<usize> = (0..n).filter(|&i| best[i]).collect();
        let p3: Vec<Vector3<f64>> = idx.iter().map(|&i| points3d[i]).collect();
        let p2: Vec<(f64, f64)> = idx.iter().map(|&i| points2d_norm[i]).collect();
        let polished = refine_pose_gauss_newton(&pose, &p3, &p2, 20);
        let polished_inliers: Vec<bool> = (0..n)
            .map(|i| reprojection_residual(&polished, &points3d[i], points2d_norm[i]) < threshold)
            .collect();
        if polished_inliers.iter().filter(|&&b| b).count() >= best.iter().filter(|&&b| b).count() {
            return Some((polished, polished_inliers));
        }
        return Some((pose, best));
    }

    if n < 6 {
        return None;
    }
    let (model, inliers) = ransac(
        n,
        6,
        &params,
        0xFACADE,
        |sample| {
            let p3: Vec<_> = sample.iter().map(|&i| points3d[i]).collect();
            if !is_well_conditioned_for_dlt(&p3) {
                return None;
            }
            let p2: Vec<_> = sample.iter().map(|&i| points2d_norm[i]).collect();
            pnp_dlt(&p3, &p2)
        },
        |pose, i| reprojection_residual(pose, &points3d[i], points2d_norm[i]),
    )?;
    let original_count = inliers.iter().filter(|&&b| b).count();
    if original_count < 6 {
        return None;
    }

    // LO refit on the full inlier set: usually a strict improvement (more,
    // better-conditioned points than any single minimal sample), but the
    // *inlier set itself* can be a near-planar point cloud even when the
    // minimal RANSAC sample that found it wasn't - and linear PnP-DLT is
    // degenerate for coplanar points (see `pnp_dlt` docs). Never let a
    // "local optimization" step return a worse answer than what RANSAC
    // already found: only accept the refit if it doesn't lose inliers.
    let inlier_idx: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
    let refit: Option<(Pose, Vec<bool>)> = (|| {
        let p3: Vec<_> = inlier_idx.iter().map(|&i| points3d[i]).collect();
        let p2: Vec<_> = inlier_idx.iter().map(|&i| points2d_norm[i]).collect();
        let refined = pnp_dlt(&p3, &p2)?;
        let final_inliers: Vec<bool> = (0..n)
            .map(|i| reprojection_residual(&refined, &points3d[i], points2d_norm[i]) < threshold)
            .collect();
        Some((refined, final_inliers))
    })();

    let (mut best_pose, mut best_inliers, mut best_count) = match refit {
        Some((refined, final_inliers))
            if final_inliers.iter().filter(|&&b| b).count() >= original_count =>
        {
            let count = final_inliers.iter().filter(|&&b| b).count();
            (refined, final_inliers, count)
        }
        _ => (model, inliers, original_count),
    };

    // Nonlinear polish: alternate a Gauss-Newton reprojection-error
    // refinement over the current inlier set with re-classifying inliers
    // against the refined pose, same never-regress guard as the DLT LO
    // refit above. Two rounds is enough for this to converge in practice -
    // the pose is already close from the linear estimate, this just removes
    // the linear solver's algebraic-error bias.
    for _ in 0..2 {
        let inlier_idx: Vec<usize> = (0..n).filter(|&i| best_inliers[i]).collect();
        if inlier_idx.len() < 6 {
            break;
        }
        let p3: Vec<_> = inlier_idx.iter().map(|&i| points3d[i]).collect();
        let p2: Vec<_> = inlier_idx.iter().map(|&i| points2d_norm[i]).collect();
        let refined = refine_pose_gauss_newton(&best_pose, &p3, &p2, 10);
        let refined_inliers: Vec<bool> = (0..n)
            .map(|i| reprojection_residual(&refined, &points3d[i], points2d_norm[i]) < threshold)
            .collect();
        let refined_count = refined_inliers.iter().filter(|&&b| b).count();
        if refined_count < best_count {
            break;
        }
        let converged = refined_count == best_count;
        best_pose = refined;
        best_inliers = refined_inliers;
        best_count = refined_count;
        if converged {
            break;
        }
    }

    Some((best_pose, best_inliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_pose_from_synthetic_correspondences() {
        let true_rotation = nalgebra::UnitQuaternion::from_euler_angles(0.05, -0.1, 0.02);
        let true_translation = Vector3::new(0.2, -0.1, 0.5);
        let true_pose = Pose::from_rotation_translation(true_rotation, true_translation);

        let points3d: Vec<Vector3<f64>> = (0..30)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.5 * (t * 0.5).sin(), 0.5 * (t * 0.3).cos(), 5.0 + 0.2 * t)
            })
            .collect();
        let points2d: Vec<(f64, f64)> = points3d
            .iter()
            .map(|p| {
                let pc = true_pose.transform_point(p);
                (pc.x / pc.z, pc.y / pc.z)
            })
            .collect();

        let (pose, inliers) =
            pnp_ransac(&points3d, &points2d, 0.01, 500).expect("should recover pose");
        assert!(inliers.iter().all(|&b| b));

        let rot_error = pose.rotation.angle_to(&true_rotation);
        assert!(rot_error < 1e-4, "rotation error: {rot_error}");
        let trans_error = (pose.translation - true_translation).norm();
        assert!(trans_error < 1e-4, "translation error: {trans_error}");
    }
}
