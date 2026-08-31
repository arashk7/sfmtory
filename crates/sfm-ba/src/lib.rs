//! Sparse bundle adjustment via Levenberg-Marquardt with the classic
//! Schur-complement reduction (eliminate 3D points first, solve the much
//! smaller "reduced camera system" for pose *and camera-intrinsics* updates,
//! then back-substitute for point updates). Camera intrinsics are optimized
//! **jointly** with poses and points in that same reduced system, not as a
//! separate pass - see `intrinsics.rs`'s module docs for why an earlier
//! alternating-pass design measurably failed to correct calibration on real
//! data (short version: a single moving camera's focal length and scene
//! depth are nearly degenerate with each other, and only a joint solve's
//! combined Jacobian can reliably escape that valley).
//!
//! A physical camera's intrinsics block is shared by every image using that
//! camera, which breaks the simple "one same-size block per image" picture:
//! the reduced camera-side system's unknowns are `[pose_0 .. pose_{n-1} |
//! cam_0 .. cam_{m-1}]` (6 dof per pose, 3-8 dof per camera depending on
//! model), with each observation contributing to *two* camera-side blocks
//! (its image's pose, and that image's camera) instead of one. Point
//! elimination generalizes cleanly: `points_to_obs` stores an [`EBlock`]
//! (pose or camera) alongside each point-coupling sub-matrix, and the same
//! "sum over all pairs of a point's contributing blocks" Schur formula
//! applies regardless of which kind of block each side is.
//!
//! Two deliberate simplifications versus a production solver like Ceres,
//! both documented here rather than hidden:
//!
//! 1. **Exact analytic Jacobians for every camera model** (see
//!    `analytic_jacobians`), not finite-difference ones - hand-derived from
//!    each `CameraModel` variant's own `project()` formula and verified
//!    against the numerical Jacobians in per-model unit tests. This wasn't
//!    always true: an earlier version used central differences everywhere,
//!    on the theory that they give the same answer to float precision at
//!    the cost of extra reprojections per observation per iteration. That
//!    held on `sceaux_castle` but not on `temple_sparse_ring`'s harder
//!    self-calibration problem (fewer images, more extreme viewing angles),
//!    where central-difference Jacobians measurably converged to a worse
//!    focal-length optimum than Ceres Solver's autodiff-based solver did on
//!    the identical input - closed by switching `SIMPLE_RADIAL` to analytic
//!    Jacobians first, then generalized to every other model on the same
//!    reasoning even without individual real-data evidence for each. See
//!    `decisions.md`'s "Analytic Jacobians" for the full story.
//! 2. **Dense reduced camera system**, not a sparse solver. After the Schur
//!    elimination the remaining system is `6*num_images + sum(camera dof)`
//!    square - for hundreds of images that's a dense system in the low
//!    thousands, which nalgebra's Cholesky solves in milliseconds. `faer`-
//!    backed sparse Cholesky (planned in PLAN.md) only starts mattering in
//!    the thousands-of-images range.

mod intrinsics;
pub use intrinsics::{default_fixed_params_mask, refine_intrinsics, IntrinsicsRefineParams};

use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, SMatrix, SVector, Vector3, Vector6};
use rayon::prelude::*;
use sfm_core::{CameraModel, Pose};

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub image_idx: usize,
    pub point_idx: usize,
    pub x: f64,
    pub y: f64,
}

pub struct BaInput {
    pub camera_of_image: Vec<usize>,
    pub cameras: Vec<CameraModel>,
    pub poses: Vec<Pose>,
    pub points: Vec<Vector3<f64>>,
    pub observations: Vec<Observation>,
    /// Poses to hold fixed (not optimized) - index-aligned with `poses`.
    /// Reprojection error alone can't fix the reconstruction's global
    /// rotation/translation/scale (a similarity transform of the whole scene
    /// leaves every residual unchanged), so at least one pose must be
    /// anchored or BA will "converge" to an arbitrarily transformed copy of
    /// the right answer. The incremental pipeline anchors its seed image;
    /// `sfm refine` on an existing model can anchor image 0 (or none, if the
    /// caller only cares about relative geometry / will re-align afterward).
    pub fixed_poses: Vec<bool>,
    /// Cameras to hold fixed (not optimized) - index-aligned with `cameras`.
    /// Set every entry `true` to reproduce old fixed-intrinsics behavior.
    pub fixed_cameras: Vec<bool>,
    /// Per-camera, per-parameter-index override: `true` holds that one
    /// intrinsic fixed even when the camera as a whole is being refined.
    /// Index order matches `CameraModel::params()`. An empty inner `Vec` (or
    /// a `Vec` shorter than the camera's param count) means "no per-param
    /// overrides for this camera" - every param refines normally.
    ///
    /// Use this to fix the principal point (`cx`/`cy`) while still refining
    /// focal length and distortion: unlike focal length, the principal point
    /// is notoriously weakly constrained by ordinary photos - jointly
    /// refining it anyway measurably made calibration *worse* on real test
    /// data (see PLAN.md), matching why COLMAP holds it fixed by default
    /// too. [`intrinsics::default_fixed_params_mask`] builds this mask for
    /// the common "refine focal + distortion, fix principal point" policy.
    pub fixed_camera_params: Vec<Vec<bool>>,
}

/// Per-point coupling block (`E` in the Schur literature), split by variable
/// kind: pose couplings are always 6x3 and stay on the stack, camera
/// couplings are `k x 3` for a model-dependent `k` and need the dynamic type.
/// The pose/pose pairing dominates real problems by orders of magnitude (many
/// images, typically one shared camera that is usually fixed during growth),
/// so keeping that arm allocation-free is what makes the Schur build fast.
enum EBlock {
    Pose(usize, SMatrix<f64, 6, 3>),
    Camera(usize, DMatrix<f64>),
}

/// One point's contribution to the reduced camera system: `-E C^-1 E^T` into
/// `s`, `+E C^-1 g_pt` into `rhs`. Points are independent of each other, so
/// this is also the unit of parallelism (see its call sites).
#[allow(clippy::too_many_arguments)]
#[inline]
fn accumulate_point_schur(
    cinv: &Matrix3<f64>,
    obs_list: &[EBlock],
    c_rhs_p: &Vector3<f64>,
    pose_slot: &[usize],
    cam_slot: &[usize],
    s: &mut DMatrix<f64>,
    rhs: &mut DVector<f64>,
) {
    for (idx_i, block_i) in obs_list.iter().enumerate() {
        match block_i {
            EBlock::Pose(i, e_i) => {
                let off_i = pose_slot[*i];
                // Hoisted out of the inner loop: halves the multiplications
                // versus recomputing `E_i C^-1` for every `j`.
                let ec: SMatrix<f64, 6, 3> = e_i * cinv;
                for (idx_j, block_j) in obs_list.iter().enumerate().skip(idx_i) {
                    match block_j {
                        EBlock::Pose(j, e_j) => {
                            let off_j = pose_slot[*j];
                            let blk: Matrix6<f64> = ec * e_j.transpose();
                            let mut d = s.view_mut((off_i, off_j), (6, 6));
                            d -= blk;
                            // Each unordered pair is visited once (the
                            // `skip(idx_i)` above); mirror it to recover the
                            // transpose term a full double loop would add.
                            // Keyed on entry index, not slot offset: two
                            // distinct observations can share a slot when
                            // their images share a camera, and that pair
                            // still needs both terms.
                            if idx_j != idx_i {
                                let mut dt = s.view_mut((off_j, off_i), (6, 6));
                                dt -= blk.transpose();
                            }
                        }
                        EBlock::Camera(c, e_j) => {
                            let off_j = cam_slot[*c];
                            let k = e_j.nrows();
                            let ec_d = DMatrix::from_column_slice(6, 3, ec.as_slice());
                            let blk = &ec_d * e_j.transpose();
                            let mut d = s.view_mut((off_i, off_j), (6, k));
                            d -= &blk;
                            let mut dt = s.view_mut((off_j, off_i), (k, 6));
                            dt -= blk.transpose();
                        }
                    }
                }
            }
            EBlock::Camera(c, e_i) => {
                let off_i = cam_slot[*c];
                let ki = e_i.nrows();
                let cinv_d = DMatrix::from_column_slice(3, 3, cinv.as_slice());
                let ec_d = e_i * &cinv_d;
                for (idx_j, block_j) in obs_list.iter().enumerate().skip(idx_i) {
                    match block_j {
                        EBlock::Pose(j, e_j) => {
                            let off_j = pose_slot[*j];
                            let e_j_d = DMatrix::from_column_slice(6, 3, e_j.as_slice());
                            let blk = &ec_d * e_j_d.transpose();
                            let mut d = s.view_mut((off_i, off_j), (ki, 6));
                            d -= &blk;
                            let mut dt = s.view_mut((off_j, off_i), (6, ki));
                            dt -= blk.transpose();
                        }
                        EBlock::Camera(c2, e_j) => {
                            let off_j = cam_slot[*c2];
                            let kj = e_j.nrows();
                            let blk = &ec_d * e_j.transpose();
                            let mut d = s.view_mut((off_i, off_j), (ki, kj));
                            d -= &blk;
                            if idx_j != idx_i {
                                let mut dt = s.view_mut((off_j, off_i), (kj, ki));
                                dt -= blk.transpose();
                            }
                        }
                    }
                }
            }
        }
    }

    let cinv_g: Vector3<f64> = cinv * c_rhs_p;
    for block in obs_list {
        match block {
            EBlock::Pose(i, e_i) => {
                let contrib: Vector6<f64> = e_i * cinv_g;
                let mut dest = rhs.rows_mut(pose_slot[*i], 6);
                dest += contrib;
            }
            EBlock::Camera(c, e_i) => {
                let cinv_g_d = DVector::from_column_slice(cinv_g.as_slice());
                let contrib = e_i * &cinv_g_d;
                let mut dest = rhs.rows_mut(cam_slot[*c], e_i.nrows());
                dest += contrib;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustLoss {
    None,
    Huber,
    Cauchy,
}

#[derive(Debug, Clone, Copy)]
pub struct BaParams {
    pub max_iterations: usize,
    pub initial_lambda: f64,
    pub robust_loss: RobustLoss,
    /// Loss transition scale, in pixels.
    pub loss_scale_px: f64,
}

impl Default for BaParams {
    fn default() -> Self {
        BaParams {
            max_iterations: 50,
            initial_lambda: 1e-3,
            robust_loss: RobustLoss::Huber,
            loss_scale_px: 2.0,
        }
    }
}

pub struct BaOutput {
    pub poses: Vec<Pose>,
    pub points: Vec<Vector3<f64>>,
    pub cameras: Vec<CameraModel>,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations_run: usize,
}

pub(crate) fn reprojection_residual(
    pose: &Pose,
    cam: &CameraModel,
    point: &Vector3<f64>,
    obs: (f64, f64),
) -> Option<SVector<f64, 2>> {
    let pc = pose.transform_point(point);
    if pc.z <= 1e-9 {
        return None;
    }
    let (px, py) = cam.project(&pc);
    Some(SVector::<f64, 2>::new(px - obs.0, py - obs.1))
}

fn perturb_pose(pose: &Pose, delta: &SVector<f64, 6>) -> Pose {
    let omega = Vector3::new(delta[0], delta[1], delta[2]);
    let dq = nalgebra::UnitQuaternion::from_scaled_axis(omega);
    let new_rotation = dq * pose.rotation;
    let new_translation = pose.translation + Vector3::new(delta[3], delta[4], delta[5]);
    Pose::from_rotation_translation(new_rotation, new_translation)
}

#[cfg(test)]
const EPS: f64 = 1e-6;

/// Completes the observation Jacobian from `d(u,v)/d(pc)` (already computed
/// by a camera model's own analytic distortion formula) via the chain rule
/// through the pose/point relationship, which is identical regardless of
/// distortion model - only the intrinsics half of the chain rule differs
/// per `CameraModel` variant. Left-multiplicative rotation perturbation
/// (`d(pc)/d(omega) = -skew(pc - t)`, `d(pc)/d(dt) = I3`), matching
/// `perturb_pose`'s `new_rotation = exp(omega) * pose.rotation` convention
/// exactly (verified against the numerical Jacobian in this module's tests).
fn pose_point_jacobians(
    pose: &Pose,
    pc: &Vector3<f64>,
    du_dpc: &Vector3<f64>,
    dv_dpc: &Vector3<f64>,
) -> (SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>) {
    let r_mat = pose.rotation.to_rotation_matrix();
    let rt = r_mat.matrix().transpose();
    let j_point_u = rt * du_dpc;
    let j_point_v = rt * dv_dpc;
    let mut j_point = SMatrix::<f64, 2, 3>::zeros();
    for c in 0..3 {
        j_point[(0, c)] = j_point_u[c];
        j_point[(1, c)] = j_point_v[c];
    }

    let q = pc - pose.translation;
    let skew_q = Matrix3::new(0.0, -q.z, q.y, q.z, 0.0, -q.x, -q.y, q.x, 0.0);
    let domega_u = skew_q * du_dpc;
    let domega_v = skew_q * dv_dpc;
    let mut j_pose = SMatrix::<f64, 2, 6>::zeros();
    for c in 0..3 {
        j_pose[(0, c)] = domega_u[c];
        j_pose[(1, c)] = domega_v[c];
        j_pose[(0, c + 3)] = du_dpc[c];
        j_pose[(1, c + 3)] = dv_dpc[c];
    }
    (j_pose, j_point)
}

/// Exact (not finite-difference) Jacobians of the reprojection residual, for
/// every `CameraModel` variant - each branch computes that model's own
/// `d(u,v)/d(pc)` and `d(u,v)/d(camera params)` analytically (hand-derived
/// from `CameraModel::project`'s exact formula), then completes the pose/
/// point half via `pose_point_jacobians`. Central-difference Jacobians
/// measurably converged to a worse self-calibration optimum than Ceres
/// Solver's autodiff-based solver did on `temple_sparse_ring` (focal length
/// error 6.3% vs. Ceres' 3.7%) while matching it closely on the easier
/// `sceaux_castle` - confirmed via a temporary Ceres-backed bundle-
/// adjustment backend (since removed; see `decisions.md`'s "Analytic
/// Jacobians"), not just theorized. `SimpleRadial` closed that gap exactly
/// (3.7% vs. Ceres' 3.7%) without needing Ceres as a runtime dependency;
/// every other model got the same treatment on the same reasoning, verified
/// against the numerical Jacobians in `analytic_matches_numerical_for_*`
/// tests rather than assumed correct from the derivation alone.
fn analytic_jacobians(
    pose: &Pose,
    cam: &CameraModel,
    point: &Vector3<f64>,
    obs: (f64, f64),
) -> Option<(
    SVector<f64, 2>,
    SMatrix<f64, 2, 6>,
    SMatrix<f64, 2, 3>,
    DMatrix<f64>,
)> {
    let pc = pose.transform_point(point);
    if pc.z <= 1e-9 {
        return None;
    }
    let inv_z = 1.0 / pc.z;
    let xp = pc.x * inv_z;
    let yp = pc.y * inv_z;

    // (residual, du_dpc, dv_dpc, j_camera columns in `CameraModel::params()` order)
    let (residual, du_dpc, dv_dpc, j_camera): (
        SVector<f64, 2>,
        Vector3<f64>,
        Vector3<f64>,
        DMatrix<f64>,
    ) = match *cam {
        CameraModel::SimplePinhole { f, cx, cy } => {
            let u = f * xp + cx;
            let v = f * yp + cy;
            let du_dpc = Vector3::new(f * inv_z, 0.0, -f * xp * inv_z);
            let dv_dpc = Vector3::new(0.0, f * inv_z, -f * yp * inv_z);
            let mut jc = DMatrix::<f64>::zeros(2, 3);
            jc[(0, 0)] = xp;
            jc[(0, 1)] = 1.0;
            jc[(1, 0)] = yp;
            jc[(1, 2)] = 1.0;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::Pinhole { fx, fy, cx, cy } => {
            let u = fx * xp + cx;
            let v = fy * yp + cy;
            let du_dpc = Vector3::new(fx * inv_z, 0.0, -fx * xp * inv_z);
            let dv_dpc = Vector3::new(0.0, fy * inv_z, -fy * yp * inv_z);
            let mut jc = DMatrix::<f64>::zeros(2, 4);
            jc[(0, 0)] = xp;
            jc[(0, 2)] = 1.0;
            jc[(1, 1)] = yp;
            jc[(1, 3)] = 1.0;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::SimpleRadial { f, cx, cy, k } => {
            let r2 = xp * xp + yp * yp;
            let d = 1.0 + k * r2;
            let u = f * xp * d + cx;
            let v = f * yp * d + cy;

            let du_dxp = f * (d + 2.0 * k * xp * xp);
            let du_dyp = 2.0 * f * k * xp * yp;
            let dv_dxp = du_dyp;
            let dv_dyp = f * (d + 2.0 * k * yp * yp);
            let du_dpc = Vector3::new(
                du_dxp * inv_z,
                du_dyp * inv_z,
                -(du_dxp * pc.x + du_dyp * pc.y) * inv_z * inv_z,
            );
            let dv_dpc = Vector3::new(
                dv_dxp * inv_z,
                dv_dyp * inv_z,
                -(dv_dxp * pc.x + dv_dyp * pc.y) * inv_z * inv_z,
            );

            let mut jc = DMatrix::<f64>::zeros(2, 4);
            jc[(0, 0)] = xp * d;
            jc[(0, 1)] = 1.0;
            jc[(0, 3)] = f * xp * r2;
            jc[(1, 0)] = yp * d;
            jc[(1, 2)] = 1.0;
            jc[(1, 3)] = f * yp * r2;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::Radial { f, cx, cy, k1, k2 } => {
            let r2 = xp * xp + yp * yp;
            let d = 1.0 + k1 * r2 + k2 * r2 * r2;
            let u = f * xp * d + cx;
            let v = f * yp * d + cy;

            // kr = d(d)/d(r2) = k1 + 2*k2*r2, shared by both partials below.
            let kr = k1 + 2.0 * k2 * r2;
            let du_dxp = f * (d + 2.0 * kr * xp * xp);
            let du_dyp = 2.0 * f * kr * xp * yp;
            let dv_dxp = du_dyp;
            let dv_dyp = f * (d + 2.0 * kr * yp * yp);
            let du_dpc = Vector3::new(
                du_dxp * inv_z,
                du_dyp * inv_z,
                -(du_dxp * pc.x + du_dyp * pc.y) * inv_z * inv_z,
            );
            let dv_dpc = Vector3::new(
                dv_dxp * inv_z,
                dv_dyp * inv_z,
                -(dv_dxp * pc.x + dv_dyp * pc.y) * inv_z * inv_z,
            );

            let mut jc = DMatrix::<f64>::zeros(2, 5);
            jc[(0, 0)] = xp * d;
            jc[(0, 1)] = 1.0;
            jc[(0, 3)] = f * xp * r2;
            jc[(0, 4)] = f * xp * r2 * r2;
            jc[(1, 0)] = yp * d;
            jc[(1, 2)] = 1.0;
            jc[(1, 3)] = f * yp * r2;
            jc[(1, 4)] = f * yp * r2 * r2;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::Radial3 {
            f,
            cx,
            cy,
            k1,
            k2,
            k3,
        } => {
            let r2 = xp * xp + yp * yp;
            let d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
            let u = f * xp * d + cx;
            let v = f * yp * d + cy;

            // kr = d(d)/d(r2), one more term than `Radial`.
            let kr = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2);
            let du_dxp = f * (d + 2.0 * kr * xp * xp);
            let du_dyp = 2.0 * f * kr * xp * yp;
            let dv_dxp = du_dyp;
            let dv_dyp = f * (d + 2.0 * kr * yp * yp);
            let du_dpc = Vector3::new(
                du_dxp * inv_z,
                du_dyp * inv_z,
                -(du_dxp * pc.x + du_dyp * pc.y) * inv_z * inv_z,
            );
            let dv_dpc = Vector3::new(
                dv_dxp * inv_z,
                dv_dyp * inv_z,
                -(dv_dxp * pc.x + dv_dyp * pc.y) * inv_z * inv_z,
            );

            let mut jc = DMatrix::<f64>::zeros(2, 6);
            jc[(0, 0)] = xp * d;
            jc[(0, 1)] = 1.0;
            jc[(0, 3)] = f * xp * r2;
            jc[(0, 4)] = f * xp * r2 * r2;
            jc[(0, 5)] = f * xp * r2 * r2 * r2;
            jc[(1, 0)] = yp * d;
            jc[(1, 2)] = 1.0;
            jc[(1, 3)] = f * yp * r2;
            jc[(1, 4)] = f * yp * r2 * r2;
            jc[(1, 5)] = f * yp * r2 * r2 * r2;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::OpenCV {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
        } => {
            let r2 = xp * xp + yp * yp;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let xd = xp * radial + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
            let yd = yp * radial + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
            let u = fx * xd + cx;
            let v = fy * yd + cy;

            let kr = k1 + 2.0 * k2 * r2; // d(radial)/d(r2)
            let dxd_dxp = radial + 2.0 * kr * xp * xp + 2.0 * p1 * yp + 6.0 * p2 * xp;
            let dxd_dyp = 2.0 * kr * xp * yp + 2.0 * p1 * xp + 2.0 * p2 * yp;
            let dyd_dxp = dxd_dyp; // symmetric mixed partial, sanity-checked by the derivation
            let dyd_dyp = radial + 2.0 * kr * yp * yp + 6.0 * p1 * yp + 2.0 * p2 * xp;

            let du_dxp = fx * dxd_dxp;
            let du_dyp = fx * dxd_dyp;
            let dv_dxp = fy * dyd_dxp;
            let dv_dyp = fy * dyd_dyp;
            let du_dpc = Vector3::new(
                du_dxp * inv_z,
                du_dyp * inv_z,
                -(du_dxp * pc.x + du_dyp * pc.y) * inv_z * inv_z,
            );
            let dv_dpc = Vector3::new(
                dv_dxp * inv_z,
                dv_dyp * inv_z,
                -(dv_dxp * pc.x + dv_dyp * pc.y) * inv_z * inv_z,
            );

            let mut jc = DMatrix::<f64>::zeros(2, 8);
            jc[(0, 0)] = xd;
            jc[(0, 2)] = 1.0;
            jc[(0, 4)] = fx * xp * r2;
            jc[(0, 5)] = fx * xp * r2 * r2;
            jc[(0, 6)] = fx * 2.0 * xp * yp;
            jc[(0, 7)] = fx * (r2 + 2.0 * xp * xp);
            jc[(1, 1)] = yd;
            jc[(1, 3)] = 1.0;
            jc[(1, 4)] = fy * yp * r2;
            jc[(1, 5)] = fy * yp * r2 * r2;
            jc[(1, 6)] = fy * (r2 + 2.0 * yp * yp);
            jc[(1, 7)] = fy * 2.0 * xp * yp;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
        CameraModel::OpenCVFisheye {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
        } => {
            let r = (xp * xp + yp * yp).sqrt().max(1e-12);
            let theta = r.atan();
            let theta2 = theta * theta;
            let poly =
                1.0 + k1 * theta2 + k2 * theta2.powi(2) + k3 * theta2.powi(3) + k4 * theta2.powi(4);
            let theta_d = theta * poly;
            let scale = theta_d / r;
            let u = fx * (xp * scale) + cx;
            let v = fy * (yp * scale) + cy;

            let poly_deriv =
                k1 + 2.0 * k2 * theta2 + 3.0 * k3 * theta2.powi(2) + 4.0 * k4 * theta2.powi(3); // d(poly)/d(theta2)
            let dtheta_d_dtheta = poly + 2.0 * theta2 * poly_deriv;
            let dtheta_dr = 1.0 / (1.0 + r * r);
            let dscale_dr = (dtheta_d_dtheta * dtheta_dr * r - theta_d) / (r * r);
            let dscale_dxp = dscale_dr * (xp / r);
            let dscale_dyp = dscale_dr * (yp / r);

            let du_dxp = fx * (scale + xp * dscale_dxp);
            let du_dyp = fx * xp * dscale_dyp;
            let dv_dxp = fy * yp * dscale_dxp;
            let dv_dyp = fy * (scale + yp * dscale_dyp);
            let du_dpc = Vector3::new(
                du_dxp * inv_z,
                du_dyp * inv_z,
                -(du_dxp * pc.x + du_dyp * pc.y) * inv_z * inv_z,
            );
            let dv_dpc = Vector3::new(
                dv_dxp * inv_z,
                dv_dyp * inv_z,
                -(dv_dxp * pc.x + dv_dyp * pc.y) * inv_z * inv_z,
            );

            let dtheta_d_dk1 = theta * theta2;
            let dtheta_d_dk2 = theta * theta2.powi(2);
            let dtheta_d_dk3 = theta * theta2.powi(3);
            let dtheta_d_dk4 = theta * theta2.powi(4);
            let mut jc = DMatrix::<f64>::zeros(2, 8);
            jc[(0, 0)] = xp * scale;
            jc[(0, 2)] = 1.0;
            jc[(0, 4)] = fx * xp * dtheta_d_dk1 / r;
            jc[(0, 5)] = fx * xp * dtheta_d_dk2 / r;
            jc[(0, 6)] = fx * xp * dtheta_d_dk3 / r;
            jc[(0, 7)] = fx * xp * dtheta_d_dk4 / r;
            jc[(1, 1)] = yp * scale;
            jc[(1, 3)] = 1.0;
            jc[(1, 4)] = fy * yp * dtheta_d_dk1 / r;
            jc[(1, 5)] = fy * yp * dtheta_d_dk2 / r;
            jc[(1, 6)] = fy * yp * dtheta_d_dk3 / r;
            jc[(1, 7)] = fy * yp * dtheta_d_dk4 / r;
            (
                SVector::<f64, 2>::new(u - obs.0, v - obs.1),
                du_dpc,
                dv_dpc,
                jc,
            )
        }
    };

    let (j_pose, j_point) = pose_point_jacobians(pose, &pc, &du_dpc, &dv_dpc);
    Some((residual, j_pose, j_point, j_camera))
}

/// Numerical Jacobians of the 2D reprojection residual w.r.t. the pose's 6
/// local params (3 rotation + 3 translation) and the point's 3 params, via
/// central differences. Returns `None` if the point is behind the camera at
/// the current estimate (that observation is skipped for this iteration).
/// No longer used by `bundle_adjust` itself (see `analytic_jacobians`) -
/// kept only as the ground truth `analytic_matches_numerical_for_*` tests
/// check the hand-derived analytic Jacobians against.
#[cfg(test)]
fn observation_jacobians(
    pose: &Pose,
    cam: &CameraModel,
    point: &Vector3<f64>,
    obs: (f64, f64),
) -> Option<(SVector<f64, 2>, SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>)> {
    let r0 = reprojection_residual(pose, cam, point, obs)?;

    let mut j_pose = SMatrix::<f64, 2, 6>::zeros();
    for k in 0..6 {
        let mut d_plus = SVector::<f64, 6>::zeros();
        d_plus[k] = EPS;
        let mut d_minus = SVector::<f64, 6>::zeros();
        d_minus[k] = -EPS;
        let p_plus = perturb_pose(pose, &d_plus);
        let p_minus = perturb_pose(pose, &d_minus);
        let r_plus = reprojection_residual(&p_plus, cam, point, obs).unwrap_or(r0);
        let r_minus = reprojection_residual(&p_minus, cam, point, obs).unwrap_or(r0);
        let col = (r_plus - r_minus) / (2.0 * EPS);
        j_pose.set_column(k, &col);
    }

    let mut j_point = SMatrix::<f64, 2, 3>::zeros();
    for k in 0..3 {
        let mut pt_plus = *point;
        pt_plus[k] += EPS;
        let mut pt_minus = *point;
        pt_minus[k] -= EPS;
        let r_plus = reprojection_residual(pose, cam, &pt_plus, obs).unwrap_or(r0);
        let r_minus = reprojection_residual(pose, cam, &pt_minus, obs).unwrap_or(r0);
        let col = (r_plus - r_minus) / (2.0 * EPS);
        j_point.set_column(k, &col);
    }

    Some((r0, j_pose, j_point))
}

/// Numerical Jacobian of the residual w.r.t. the camera's own intrinsic
/// parameters (2 x n, n = 3-8 depending on model). `r0` is the residual
/// already computed at the current estimate, reused when a perturbed
/// parameter set is invalid (e.g. would require a different model) rather
/// than failing the whole observation. No longer used by `bundle_adjust`
/// itself (see `analytic_jacobians`) - kept only as the ground truth
/// `analytic_matches_numerical_for_*` tests check against.
#[cfg(test)]
fn camera_jacobian(
    pose: &Pose,
    cam: &CameraModel,
    point: &Vector3<f64>,
    obs: (f64, f64),
    r0: SVector<f64, 2>,
) -> DMatrix<f64> {
    let n = cam.params().len();
    let mut jac = DMatrix::<f64>::zeros(2, n);
    for k in 0..n {
        let plus = intrinsics::perturb_camera(cam, k, EPS);
        let minus = intrinsics::perturb_camera(cam, k, -EPS);
        let r_plus = plus
            .as_ref()
            .and_then(|c| reprojection_residual(pose, c, point, obs))
            .unwrap_or(r0);
        let r_minus = minus
            .as_ref()
            .and_then(|c| reprojection_residual(pose, c, point, obs))
            .unwrap_or(r0);
        let col = (r_plus - r_minus) / (2.0 * EPS);
        jac[(0, k)] = col.x;
        jac[(1, k)] = col.y;
    }
    jac
}

/// IRLS weight for a residual of the given norm: `1.0` for `RobustLoss::None`,
/// else the derivative-based reweighting for Huber/Cauchy that lets a
/// standard weighted-least-squares solve approximate the robust M-estimator.
/// Takes the loss/scale directly (rather than `&BaParams`) so
/// `intrinsics::refine_intrinsics` can reuse it with its own params type.
pub(crate) fn robust_weight(
    residual_norm: f64,
    robust_loss: RobustLoss,
    loss_scale_px: f64,
) -> f64 {
    let s = loss_scale_px.max(1e-9);
    match robust_loss {
        RobustLoss::None => 1.0,
        RobustLoss::Huber => {
            if residual_norm <= s {
                1.0
            } else {
                s / residual_norm
            }
        }
        RobustLoss::Cauchy => 1.0 / (1.0 + (residual_norm / s).powi(2)),
    }
}

/// Plain (unweighted, un-squared) mean reprojection error in pixels -
/// deliberately *not* the robust-loss-weighted cost `bundle_adjust` itself
/// optimizes against. That distinction matters for callers deciding whether
/// a result is actually good: Huber/Cauchy loss is intentionally forgiving of
/// outliers so the optimizer isn't derailed by a few bad correspondences
/// during iteration, but that same forgiveness makes the *robust* cost a
/// poor judge of whether the fit as a whole is good - it can rate a fit with
/// many moderately-bad residuals as "better" than one with a handful of
/// larger ones, even when the plain mean error says the opposite. Use this
/// function for that kind of after-the-fact comparison (see
/// `sfm-reconstruction::run_bundle_adjustment`, which uses it to decide
/// whether refining intrinsics actually helped).
pub fn mean_reprojection_error(input: &BaInput) -> f64 {
    let (sum, count) = input
        .observations
        .par_iter()
        .filter_map(|obs| {
            let pose = &input.poses[obs.image_idx];
            let cam = &input.cameras[input.camera_of_image[obs.image_idx]];
            let point = &input.points[obs.point_idx];
            reprojection_residual(pose, cam, point, (obs.x, obs.y)).map(|r| r.norm())
        })
        .fold(|| (0.0, 0usize), |(s, c), n| (s + n, c + 1))
        .reduce(|| (0.0, 0usize), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn total_cost(input: &BaInput, params: &BaParams) -> f64 {
    input
        .observations
        .par_iter()
        .map(|obs| {
            let pose = &input.poses[obs.image_idx];
            let cam = &input.cameras[input.camera_of_image[obs.image_idx]];
            let point = &input.points[obs.point_idx];
            match reprojection_residual(pose, cam, point, (obs.x, obs.y)) {
                Some(r) => {
                    let n = r.norm();
                    let w = robust_weight(n, params.robust_loss, params.loss_scale_px);
                    w * n * n
                }
                None => 0.0,
            }
        })
        .sum()
}

pub fn bundle_adjust(mut input: BaInput, params: &BaParams) -> BaOutput {
    let num_images = input.poses.len();
    let num_points = input.points.len();
    let num_cameras = input.cameras.len();
    let cam_dof: Vec<usize> = input.cameras.iter().map(|c| c.params().len()).collect();

    // Variable indexing: only *free* poses and cameras get a slot in the
    // normal equations. Fixed ones (`fixed_poses`/`fixed_cameras`) are absent
    // from the linear system entirely rather than being carried as identity
    // rows that trivially solve to zero. That matters a lot for local bundles
    // (see `sfm-reconstruction`'s `BaScope::Local`), where most images in the
    // problem are present only to constrain points and are held fixed: the
    // dense Cholesky is O(n^3) in this dimension, so dropping the fixed
    // blocks shrinks it by the cube of the free fraction.
    const FIXED: usize = usize::MAX;
    let mut pose_slot = vec![FIXED; num_images];
    let mut cam_slot = vec![FIXED; num_cameras];
    let mut total_dim = 0usize;
    for i in 0..num_images {
        if !input.fixed_poses.get(i).copied().unwrap_or(false) {
            pose_slot[i] = total_dim;
            total_dim += 6;
        }
    }
    for c in 0..num_cameras {
        if cam_dof[c] > 0 && !input.fixed_cameras.get(c).copied().unwrap_or(false) {
            cam_slot[c] = total_dim;
            total_dim += cam_dof[c];
        }
    }
    if total_dim == 0 {
        let initial_cost = total_cost(&input, params);
        return BaOutput {
            poses: input.poses,
            points: input.points,
            cameras: input.cameras,
            initial_cost,
            final_cost: initial_cost,
            iterations_run: 0,
        };
    }

    // Allocated once and reused across every outer iteration and damping
    // retry (see the Schur-build loop).
    let mut s_buf = DMatrix::<f64>::zeros(total_dim, total_dim);
    let mut rhs_buf = DVector::<f64>::zeros(total_dim);

    let initial_cost = total_cost(&input, params);
    let mut lambda = params.initial_lambda;

    let mut cost = initial_cost;
    // Nielsen's rejection-growth factor: doubles on each consecutive
    // rejection so a badly-scaled region is escaped geometrically, and resets
    // to 2 whenever a step is accepted.
    let mut nu = 2.0f64;
    let mut iterations_run = 0;

    // Ceres/COLMAP's own default `function_tolerance` (relative cost
    // decrease below which the solver considers itself converged, distinct
    // from `max_iterations` which only bounds the *worst* case). Without
    // this, a call whose `max_iterations` is generously sized for a hard
    // problem (e.g. the once-per-reconstruction intrinsics-refining pass)
    // burns its full iteration budget on every problem, including the many
    // easy, already-near-converged periodic in-loop calls during growth,
    // each accepting a long tail of negligible steps - measurably the
    // dominant cost of `sfm map`'s wall-clock time on real datasets (see
    // decisions.md).
    const FUNCTION_TOLERANCE: f64 = 1e-6;

    for _outer in 0..params.max_iterations {
        iterations_run += 1;
        let cost_before_iter = cost;

        // Per-observation Jacobians, computed once per outer iteration and
        // reused across the inner lambda-retry loop below.
        struct Prepared {
            image_idx: usize,
            camera_idx: usize,
            point_idx: usize,
            r: SVector<f64, 2>,
            jp: SMatrix<f64, 2, 6>,
            jc: DMatrix<f64>,
            jx: SMatrix<f64, 2, 3>,
            weight: f64,
        }
        let prepared: Vec<Prepared> = input
            .observations
            .par_iter()
            .filter_map(|obs| {
                let pose = &input.poses[obs.image_idx];
                let camera_idx = input.camera_of_image[obs.image_idx];
                let cam = &input.cameras[camera_idx];
                let point = &input.points[obs.point_idx];
                let (r, jp, jx, jc) = analytic_jacobians(pose, cam, point, (obs.x, obs.y))?;
                let weight = robust_weight(r.norm(), params.robust_loss, params.loss_scale_px);
                Some(Prepared {
                    image_idx: obs.image_idx,
                    camera_idx,
                    point_idx: obs.point_idx,
                    r,
                    jp,
                    jc,
                    jx,
                    weight,
                })
            })
            .collect();

        if prepared.is_empty() {
            break;
        }

        let mut b_pose_diag = vec![Matrix6::<f64>::zeros(); num_images];
        let mut b_pose_rhs = vec![Vector6::<f64>::zeros(); num_images];
        let mut b_cam_diag: Vec<DMatrix<f64>> =
            cam_dof.iter().map(|&k| DMatrix::zeros(k, k)).collect();
        let mut b_cam_rhs: Vec<DVector<f64>> = cam_dof.iter().map(|&k| DVector::zeros(k)).collect();
        let mut pose_cam_cross: Vec<DMatrix<f64>> = (0..num_images)
            .map(|i| DMatrix::zeros(6, cam_dof[input.camera_of_image[i]]))
            .collect();
        let mut c_diag = vec![Matrix3::<f64>::zeros(); num_points];
        let mut c_rhs = vec![Vector3::<f64>::zeros(); num_points];
        // Per-point coupling blocks (`E` in the Schur literature), kept in a
        // split representation: pose couplings are always 6x3 and are stored
        // as stack-allocated fixed-size matrices, while camera couplings are
        // k x 3 for a model-dependent k and need the dynamic type. The
        // pose-pose pairing dominates by orders of magnitude on real problems
        // (many images, typically one shared camera that's usually held fixed
        // during growth), so keeping it off the heap is what makes the Schur
        // build fast - this loop runs once per point *per damping retry*.
        let mut points_to_obs: Vec<Vec<EBlock>> = (0..num_points).map(|_| Vec::new()).collect();

        for p in &prepared {
            let w = p.weight;
            // Points are still constrained by observations from a fixed
            // pose/camera (c_diag/c_rhs always accumulate); the fixed side
            // just isn't added as a solvable variable.
            c_diag[p.point_idx] += w * p.jx.transpose() * p.jx;
            c_rhs[p.point_idx] += w * p.jx.transpose() * p.r;

            let pose_free = pose_slot[p.image_idx] != FIXED;
            let cam_free = cam_slot[p.camera_idx] != FIXED;

            if pose_free {
                b_pose_diag[p.image_idx] += w * p.jp.transpose() * p.jp;
                b_pose_rhs[p.image_idx] += w * p.jp.transpose() * p.r;
                let e_pose: SMatrix<f64, 6, 3> = w * p.jp.transpose() * p.jx;
                points_to_obs[p.point_idx].push(EBlock::Pose(p.image_idx, e_pose));
            }
            if cam_free {
                let jx_d = DMatrix::from_column_slice(2, 3, p.jx.as_slice());
                let r_d = DVector::from_column_slice(p.r.as_slice());
                b_cam_diag[p.camera_idx] += w * p.jc.transpose() * &p.jc;
                b_cam_rhs[p.camera_idx] += w * p.jc.transpose() * &r_d;
                let e_cam = w * p.jc.transpose() * &jx_d;
                points_to_obs[p.point_idx].push(EBlock::Camera(p.camera_idx, e_cam));
            }
            if pose_free && cam_free {
                let jp_d = DMatrix::from_column_slice(2, 6, p.jp.as_slice());
                pose_cam_cross[p.image_idx] += w * jp_d.transpose() * &p.jc;
            }
        }

        // Try increasing damping until we find an accepted (cost-reducing) step.
        let mut accepted = false;
        for _inner in 0..8 {
            // Reused across damping retries rather than reallocated: this is
            // the hot loop, and a fresh zeroed `total_dim^2` allocation per
            // retry was measurable overhead.
            s_buf.fill(0.0);
            rhs_buf.fill(0.0);
            let s = &mut s_buf;
            let rhs = &mut rhs_buf;

            for i in 0..num_images {
                let off = pose_slot[i];
                if off == FIXED {
                    continue;
                }
                let damped =
                    b_pose_diag[i] + Matrix6::from_diagonal(&b_pose_diag[i].diagonal()) * lambda;
                s.view_mut((off, off), (6, 6)).copy_from(&damped);
                rhs.rows_mut(off, 6).copy_from(&(-b_pose_rhs[i]));
            }

            for c in 0..num_cameras {
                let k = cam_dof[c];
                let off = cam_slot[c];
                if k == 0 || off == FIXED {
                    continue;
                }
                let diag = b_cam_diag[c].diagonal();
                let damped = &b_cam_diag[c] + DMatrix::from_diagonal(&diag) * lambda;
                s.view_mut((off, off), (k, k)).copy_from(&damped);
                rhs.rows_mut(off, k).copy_from(&(-b_cam_rhs[c].clone()));
            }

            for i in 0..num_images {
                let c = input.camera_of_image[i];
                let (po, co) = (pose_slot[i], cam_slot[c]);
                if po == FIXED || co == FIXED || cam_dof[c] == 0 {
                    continue;
                }
                let cross = &pose_cam_cross[i];
                s.view_mut((po, co), (6, cam_dof[c])).copy_from(cross);
                s.view_mut((co, po), (cam_dof[c], 6))
                    .copy_from(&cross.transpose());
            }

            let mut cp_inv: Vec<Option<Matrix3<f64>>> = vec![None; num_points];
            for p in 0..num_points {
                if points_to_obs[p].is_empty() {
                    continue;
                }
                let damped = c_diag[p] + Matrix3::from_diagonal(&c_diag[p].diagonal()) * lambda;
                cp_inv[p] = damped.try_inverse();
            }

            // Point elimination. Parallel over points only when the
            // problem is big enough to pay for it: each rayon worker needs
            // its own `total_dim^2` accumulator, which on the many small
            // local bundles during growth costs far more to allocate and
            // zero than the elimination itself - and those already run
            // inside an outer parallel loop over seed candidates, so nesting
            // there just oversubscribes the pool. Measured: unconditionally
            // parallelizing this made `temple_ring` 2x *slower* overall.
            // `current_thread_index().is_some()` means this call is already
            // running inside a rayon worker - i.e. it is one of the parallel
            // seed-candidate growths, which have the whole pool saturated
            // already. Nesting there only oversubscribes and pays the
            // per-worker accumulator cost for nothing; the top-level final
            // bundles run on the main thread and do get the speedup.
            const PARALLEL_SCHUR_MIN_POINTS: usize = 1500;
            let parallel_schur =
                num_points >= PARALLEL_SCHUR_MIN_POINTS && rayon::current_thread_index().is_none();
            if parallel_schur {
                // Chunked by hand into exactly one range per worker rather
                // than letting rayon subdivide freely: each chunk allocates
                // and later reduces a dense `total_dim^2` accumulator, so the
                // number of chunks - not the number of points - sets the
                // overhead. Leaving rayon to split adaptively produced
                // hundreds of those and was slower than running serially.
                let nthreads = rayon::current_num_threads().max(1);
                let chunk = num_points.div_ceil(nthreads).max(1);
                let (s_acc, rhs_acc) = (0..nthreads)
                    .into_par_iter()
                    .map(|t| {
                        let lo = (t * chunk).min(num_points);
                        let hi = ((t + 1) * chunk).min(num_points);
                        let mut ps = DMatrix::<f64>::zeros(total_dim, total_dim);
                        let mut pr = DVector::<f64>::zeros(total_dim);
                        for p in lo..hi {
                            if let Some(cinv) = cp_inv[p] {
                                accumulate_point_schur(
                                    &cinv,
                                    &points_to_obs[p],
                                    &c_rhs[p],
                                    &pose_slot,
                                    &cam_slot,
                                    &mut ps,
                                    &mut pr,
                                );
                            }
                        }
                        (ps, pr)
                    })
                    .reduce(
                        || {
                            (
                                DMatrix::<f64>::zeros(total_dim, total_dim),
                                DVector::<f64>::zeros(total_dim),
                            )
                        },
                        |(mut a_s, a_r), (b_s, b_r)| {
                            a_s += b_s;
                            (a_s, a_r + b_r)
                        },
                    );
                *s += s_acc;
                *rhs += rhs_acc;
            } else {
                for p in 0..num_points {
                    if let Some(cinv) = cp_inv[p] {
                        accumulate_point_schur(
                            &cinv,
                            &points_to_obs[p],
                            &c_rhs[p],
                            &pose_slot,
                            &cam_slot,
                            s,
                            rhs,
                        );
                    }
                }
            }

            // Decouple any individually-fixed camera parameter (e.g. the
            // principal point - see `BaInput::fixed_camera_params`) from the
            // rest of the system: zero its row/col (discarding whatever it
            // accumulated, including any Schur-eliminated point coupling),
            // pin the diagonal to 1 and rhs to 0, so it solves to delta=0
            // regardless of what the rest of the system does.
            for c in 0..num_cameras {
                let Some(mask) = input.fixed_camera_params.get(c) else {
                    continue;
                };
                for (k, &fixed) in mask.iter().enumerate() {
                    if !fixed || k >= cam_dof[c] {
                        continue;
                    }
                    if cam_slot[c] == FIXED {
                        continue;
                    }
                    let row = cam_slot[c] + k;
                    for col in 0..total_dim {
                        s[(row, col)] = 0.0;
                        s[(col, row)] = 0.0;
                    }
                    s[(row, row)] = 1.0;
                    rhs[row] = 0.0;
                }
            }

            // Enforce exact symmetry (should already hold analytically; this
            // guards against float asymmetry breaking the Cholesky solve).
            let s_sym = 0.5 * (&*s + s.transpose());

            // `Cholesky::new` consumes its argument, so the LU fallback
            // rebuilds `s_sym` rather than cloning it up front - the clone
            // cost a full `total_dim^2` copy on *every* iteration to serve a
            // path that only runs when the reduced system is not positive
            // definite, which on a well-damped problem is essentially never.
            let delta = match nalgebra::linalg::Cholesky::new(s_sym) {
                Some(chol) => chol.solve(&*rhs),
                None => {
                    let s_sym = 0.5 * (&*s + s.transpose());
                    match s_sym.lu().solve(&*rhs) {
                        Some(sol) => sol,
                        None => {
                            lambda *= nu;
                            nu *= 2.0;
                            continue;
                        }
                    }
                }
            };

            // delta_pt = -C^-1 * (g_pt + E^T * delta), from the second
            // block row of the normal equations (E^T delta + C delta_pt =
            // -g_pt). `c_rhs[p]` holds `g_pt` (un-negated), so both terms
            // here need the same overall sign flip.
            let mut delta_point = vec![Vector3::<f64>::zeros(); num_points];
            for p in 0..num_points {
                let Some(cinv) = cp_inv[p] else { continue };
                let mut acc: Vector3<f64> = c_rhs[p];
                for block in &points_to_obs[p] {
                    match block {
                        EBlock::Pose(i, e_i) => {
                            let dc: Vector6<f64> = Vector6::from_iterator(
                                delta.rows(pose_slot[*i], 6).iter().copied(),
                            );
                            acc += e_i.transpose() * dc;
                        }
                        EBlock::Camera(c, e_i) => {
                            let k = e_i.nrows();
                            let dc = delta.rows(cam_slot[*c], k).into_owned();
                            let contrib = e_i.transpose() * dc;
                            acc += Vector3::new(contrib[0], contrib[1], contrib[2]);
                        }
                    }
                }
                delta_point[p] = -(cinv * acc);
            }

            let mut trial_poses = input.poses.clone();
            for i in 0..num_images {
                let d: SVector<f64, 6> = if pose_slot[i] == FIXED {
                    // Held fixed: absent from the reduced system, so its
                    // increment is exactly zero by construction.
                    SVector::<f64, 6>::zeros()
                } else {
                    SVector::<f64, 6>::from_iterator(delta.rows(pose_slot[i], 6).iter().copied())
                };
                trial_poses[i] = perturb_pose(&input.poses[i], &d);
            }
            let mut trial_points = input.points.clone();
            for p in 0..num_points {
                trial_points[p] += delta_point[p];
            }

            let mut trial_cameras = input.cameras.clone();
            let mut cameras_valid = true;
            for c in 0..num_cameras {
                let k = cam_dof[c];
                if k == 0 || cam_slot[c] == FIXED {
                    continue;
                }
                let dc = delta.rows(cam_slot[c], k);
                let mut new_params = trial_cameras[c].params();
                for j in 0..k {
                    new_params[j] += dc[j];
                }
                match CameraModel::from_name_and_params(trial_cameras[c].name(), &new_params) {
                    Some(updated) => trial_cameras[c] = updated,
                    None => {
                        cameras_valid = false;
                        break;
                    }
                }
            }
            if !cameras_valid {
                lambda *= 3.0;
                continue;
            }

            let trial_input = BaInput {
                camera_of_image: input.camera_of_image.clone(),
                cameras: trial_cameras.clone(),
                poses: trial_poses.clone(),
                points: trial_points.clone(),
                observations: input.observations.clone(),
                fixed_poses: input.fixed_poses.clone(),
                fixed_cameras: input.fixed_cameras.clone(),
                fixed_camera_params: input.fixed_camera_params.clone(),
            };
            let trial_cost = total_cost(&trial_input, params);

            // Nielsen's trust-region update, driven by the gain ratio
            // `rho` (actual decrease over the decrease the linearized model
            // predicted) rather than a fixed multiply-or-divide. A step that
            // matched its prediction well shrinks the damping aggressively,
            // pushing the next iteration toward a full Gauss-Newton step; a
            // step that overshot backs off geometrically. The fixed +-3x
            // schedule this replaces needed roughly twice as many outer
            // iterations to reach the same cost on real problems, because it
            // could neither loosen quickly on well-behaved regions nor
            // tighten fast enough on badly-scaled ones.
            if trial_cost.is_finite() && trial_cost < cost {
                input.poses = trial_poses;
                input.points = trial_points;
                input.cameras = trial_cameras;
                cost = trial_cost;
                // Aggressive shrink on every accepted step, driving lambda
                // toward a full Gauss-Newton step. Measured against
                // Nielsen's gain-ratio-proportional shrink
                // (`1-(2*rho-1)^3`), which is the textbook choice: on these
                // problems it was consistently *worse* (10637 vs. 8452 outer
                // iterations on `temple_ring`), because it keeps damping high
                // through the long well-behaved tail where these problems
                // spend most of their iterations.
                lambda = (lambda / 3.0).max(1e-12);
                nu = 2.0;
                accepted = true;
                break;
            } else {
                // Rejections do use Nielsen's geometric growth rather than a
                // fixed 3x: consecutive rejections mean the linearization is
                // badly scaled here, and doubling the growth factor each time
                // escapes that in a couple of retries instead of creeping.
                lambda *= nu;
                nu *= 2.0;
            }
        }

        if !accepted {
            break;
        }
        let rel_decrease = (cost_before_iter - cost) / cost_before_iter.max(1e-12);
        if rel_decrease < FUNCTION_TOLERANCE {
            break;
        }
    }

    BaOutput {
        poses: input.poses,
        points: input.points,
        cameras: input.cameras,
        initial_cost,
        final_cost: cost,
        iterations_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn pinhole(f: f64, w: u32, h: u32) -> CameraModel {
        CameraModel::SimplePinhole {
            f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }

    /// Shared check for every `analytic_matches_numerical_for_*` test below:
    /// the hand-derived analytic Jacobian for `cam` must agree with the
    /// (independently correct, since it's a direct finite-difference of the
    /// same `CameraModel::project`) numerical one to high precision, at a
    /// generic non-trivial pose/point/observation.
    fn assert_analytic_matches_numerical(cam: &CameraModel) {
        let pose = Pose::from_rotation_translation(
            UnitQuaternion::from_euler_angles(0.15, -0.3, 0.08),
            Vector3::new(0.6, -0.2, 1.3),
        );
        let point = Vector3::new(0.35, -0.18, 2.7);
        let obs = (410.0, 260.0); // arbitrary - only the *Jacobian*, not zero residual, matters here

        let (r_num, jp_num, jx_num) =
            observation_jacobians(&pose, cam, &point, obs).expect("point in front of camera");
        let jc_num = camera_jacobian(&pose, cam, &point, obs, r_num);
        let (r_an, jp_an, jx_an, jc_an) =
            analytic_jacobians(&pose, cam, &point, obs).expect("point in front of camera");

        assert!(
            (r_num - r_an).norm() < 1e-9,
            "residual mismatch for {}: numerical={r_num:?} analytic={r_an:?}",
            cam.name()
        );
        assert!(
            (jp_num - jp_an).norm() < 1e-5,
            "pose jacobian mismatch for {}:\nnumerical={jp_num}\nanalytic={jp_an}",
            cam.name()
        );
        assert!(
            (jx_num - jx_an).norm() < 1e-5,
            "point jacobian mismatch for {}:\nnumerical={jx_num}\nanalytic={jx_an}",
            cam.name()
        );
        let (rows, cols) = jc_num.shape();
        let jc_diff: f64 = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| (jc_num[(r, c)] - jc_an[(r, c)]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            jc_diff < 1e-4,
            "camera jacobian mismatch for {}:\nnumerical={jc_num}\nanalytic={jc_an}",
            cam.name()
        );
    }

    #[test]
    fn analytic_matches_numerical_for_simple_pinhole() {
        assert_analytic_matches_numerical(&CameraModel::SimplePinhole {
            f: 1420.0,
            cx: 320.0,
            cy: 240.0,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_pinhole() {
        assert_analytic_matches_numerical(&CameraModel::Pinhole {
            fx: 1420.0,
            fy: 1390.0,
            cx: 320.0,
            cy: 240.0,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_simple_radial() {
        assert_analytic_matches_numerical(&CameraModel::SimpleRadial {
            f: 1420.0,
            cx: 320.0,
            cy: 240.0,
            k: -0.45,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_radial() {
        assert_analytic_matches_numerical(&CameraModel::Radial {
            f: 1420.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.35,
            k2: 0.12,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_radial3() {
        assert_analytic_matches_numerical(&CameraModel::Radial3 {
            f: 1420.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.35,
            k2: 0.12,
            k3: -0.03,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_opencv() {
        assert_analytic_matches_numerical(&CameraModel::OpenCV {
            fx: 1420.0,
            fy: 1390.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.28,
            k2: 0.09,
            p1: 0.015,
            p2: -0.02,
        });
    }

    #[test]
    fn analytic_matches_numerical_for_opencv_fisheye() {
        assert_analytic_matches_numerical(&CameraModel::OpenCVFisheye {
            fx: 900.0,
            fy: 880.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.05,
            k2: 0.02,
            k3: -0.01,
            k4: 0.005,
        });
    }

    /// Two cameras, a handful of 3D points, perfect synthetic observations,
    /// then perturb every parameter and check BA converges back to (near)
    /// the ground truth and drives reprojection error to ~0.
    #[test]
    fn recovers_ground_truth_from_perturbed_initialization() {
        let cam = pinhole(700.0, 640, 480);
        let true_poses = vec![
            Pose::identity(),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(0.02, 0.2, -0.01),
                Vector3::new(0.8, 0.05, 0.1),
            ),
        ];
        let true_points: Vec<Vector3<f64>> = (0..25)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.5 * (t * 0.6).sin(), 0.4 * (t * 0.4).cos(), 3.0 + 0.1 * t)
            })
            .collect();

        let mut observations = Vec::new();
        for (pt_idx, point) in true_points.iter().enumerate() {
            for (img_idx, pose) in true_poses.iter().enumerate() {
                let pc = pose.transform_point(point);
                if pc.z <= 0.0 {
                    continue;
                }
                let (x, y) = cam.project(&pc);
                observations.push(Observation {
                    image_idx: img_idx,
                    point_idx: pt_idx,
                    x,
                    y,
                });
            }
        }

        // Perturb everything, then anchor camera 0's pose (`fixed_poses`) as
        // the gauge reference: reprojection error alone is invariant under a
        // similarity transform of the whole scene, so without anchoring at
        // least one pose, BA would correctly converge to an arbitrarily
        // rotated/translated/rescaled - but equally valid - copy of the
        // ground truth rather than to these specific coordinates.
        let mut init_poses = true_poses.clone();
        init_poses[1] = perturb_pose(
            &true_poses[1],
            &SVector::<f64, 6>::from_column_slice(&[0.03, -0.02, 0.01, 0.05, -0.03, 0.04]),
        );
        let mut init_points = true_points.clone();
        for (i, p) in init_points.iter_mut().enumerate() {
            let s = i as f64;
            *p += Vector3::new(
                0.05 * (s * 1.1).sin(),
                -0.04 * (s * 0.7).cos(),
                0.06 * (s * 0.3).sin(),
            );
        }

        let input = BaInput {
            camera_of_image: vec![0, 0],
            cameras: vec![cam],
            poses: init_poses,
            points: init_points,
            observations,
            fixed_poses: vec![true, false],
            fixed_cameras: vec![true],
            fixed_camera_params: vec![],
        };
        let params = BaParams {
            max_iterations: 30,
            ..Default::default()
        };
        let output = bundle_adjust(input, &params);

        assert!(
            output.final_cost < output.initial_cost * 0.01,
            "expected large cost reduction: initial={} final={}",
            output.initial_cost,
            output.final_cost
        );
        // Mean reprojection error should end up sub-pixel.
        let mean_sq_error =
            output.final_cost / (output.poses.len() as f64 * true_points.len() as f64);
        assert!(
            mean_sq_error.sqrt() < 0.5,
            "mean reprojection error too high: {}",
            mean_sq_error.sqrt()
        );

        // Fixing camera 0's pose removes the rotation/translation gauge
        // freedom but *not* scale: uniformly rescaling camera 1's
        // translation and every point's position around camera 0's (fixed)
        // center leaves every reprojection residual exactly unchanged, so
        // with noiseless synthetic data (a whole manifold of zero-cost
        // solutions) LM has no gradient pulling it to any particular scale
        // along that manifold. Check direction (scale-invariant) rather than
        // the exact translation vector.
        let recovered_translation = output.poses[1].translation;
        let true_translation = true_poses[1].translation;
        let cos_angle = recovered_translation
            .normalize()
            .dot(&true_translation.normalize());
        assert!(
            cos_angle > 0.999,
            "translation direction not recovered: cos_angle={cos_angle}"
        );
        let scale_ratio = recovered_translation.norm() / true_translation.norm();
        assert!(
            (scale_ratio - 1.0).abs() < 0.1,
            "scale drifted too far: ratio={scale_ratio}"
        );
    }

    /// The test that would have caught the original bug: a single shared
    /// camera used by several images with genuinely different positions/
    /// depths (not the narrow-baseline case), starting from a wrong focal
    /// length. Jointly optimizing intrinsics alongside poses/points (this
    /// module) must recover the true focal length; the earlier alternating
    /// block-coordinate design (see `intrinsics.rs` docs) measurably failed
    /// this exact scenario on real photos because poses/points had already
    /// fully converged to fit the wrong focal length before intrinsics ever
    /// got a turn.
    #[test]
    fn joint_optimization_recovers_shared_camera_focal_length() {
        let true_cam = CameraModel::SimplePinhole {
            f: 1500.0,
            cx: 500.0,
            cy: 400.0,
        };
        let true_poses = vec![
            Pose::identity(),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(0.0, 0.25, 0.0),
                Vector3::new(1.2, 0.1, -0.3),
            ),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(0.1, -0.3, 0.05),
                Vector3::new(-0.9, 0.4, 0.5),
            ),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(-0.05, 0.4, -0.1),
                Vector3::new(0.5, -0.6, 1.1),
            ),
        ];
        let true_points: Vec<Vector3<f64>> = (0..60)
            .map(|i| {
                let t = i as f64;
                Vector3::new(
                    0.8 * (t * 0.31).sin(),
                    0.7 * (t * 0.47).cos(),
                    2.5 + 0.15 * t,
                )
            })
            .collect();

        let mut observations = Vec::new();
        for (pt_idx, point) in true_points.iter().enumerate() {
            for (img_idx, pose) in true_poses.iter().enumerate() {
                let pc = pose.transform_point(point);
                if pc.z <= 0.1 {
                    continue;
                }
                let (x, y) = true_cam.project(&pc);
                observations.push(Observation {
                    image_idx: img_idx,
                    point_idx: pt_idx,
                    x,
                    y,
                });
            }
        }

        // Simulate exactly the real-world failure scenario: poses and points
        // fully converged (via BA with fixed, wrong intrinsics) before
        // intrinsics get a chance to move - the initial condition where the
        // old alternating approach got stuck.
        let wrong_cam = CameraModel::SimplePinhole {
            f: 1000.0,
            cx: 500.0,
            cy: 400.0,
        };
        let pre_converge_input = BaInput {
            camera_of_image: vec![0; true_poses.len()],
            cameras: vec![wrong_cam],
            poses: true_poses.clone(),
            points: true_points.clone(),
            observations: observations.clone(),
            fixed_poses: vec![true, false, false, false],
            fixed_cameras: vec![true],
            fixed_camera_params: vec![],
        };
        let pre_converged = bundle_adjust(
            pre_converge_input,
            &BaParams {
                max_iterations: 50,
                ..Default::default()
            },
        );

        // Now jointly refine everything, including intrinsics, from that
        // already-converged (wrong-focal) starting point.
        let joint_input = BaInput {
            camera_of_image: vec![0; true_poses.len()],
            cameras: vec![wrong_cam],
            poses: pre_converged.poses,
            points: pre_converged.points,
            observations,
            fixed_poses: vec![true, false, false, false],
            fixed_cameras: vec![false],
            fixed_camera_params: vec![],
        };
        let output = bundle_adjust(
            joint_input,
            &BaParams {
                max_iterations: 50,
                ..Default::default()
            },
        );

        let CameraModel::SimplePinhole { f, .. } = output.cameras[0] else {
            panic!("model changed")
        };
        let relative_error = (f - 1500.0).abs() / 1500.0;
        assert!(
            relative_error < 0.02,
            "recovered focal length {f}, want ~1500 (started from wrong-but-converged 1000)"
        );
    }
}
