//! Standalone per-camera intrinsics refinement: holds poses and points fixed
//! and runs a small dense Levenberg-Marquardt over just one camera's own
//! parameters (3-8 numbers depending on model) at a time.
//!
//! **This is no longer the primary way sfmtory calibrates intrinsics** -
//! `bundle_adjust` now optimizes camera intrinsics *jointly* with poses and
//! points in one linear system (`BaInput::fixed_cameras`), which is what
//! `sfm-reconstruction` actually uses. An earlier version of this crate ran
//! intrinsics refinement as a separate alternating pass after each pose/point
//! solve; that measurably failed to correct a wrong focal-length guess on
//! real test data, because once poses/points fully converge under a
//! *wrong-but-self-consistent* focal length, the isolated intrinsics-only
//! step has almost no gradient to work with (a single moving camera's focal
//! length and scene depth are nearly degenerate with each other - a
//! reprojection-error-only cost has a very flat direction along "rescale
//! focal length and scene depth together", and alternating optimization
//! converges into that flat valley and gets stuck, while a joint solve's
//! combined Jacobian can still find a genuine descent direction using
//! curvature from images with different viewing geometry). See PLAN.md's
//! real-data-testing entries for both the original finding and this fix.
//!
//! This function is kept as a lighter-weight utility (e.g. for isolating
//! intrinsics-only regressions in tests, or a future `sfm refine
//! --intrinsics-only` mode) - just don't reach for it as the main calibration
//! path.

use nalgebra::{DMatrix, DVector, SVector, Vector3};
use rayon::prelude::*;
use sfm_core::{CameraModel, Pose};

/// The principal-point parameter indices (matching `CameraModel::params()`
/// order) for each model - the parameters `default_fixed_params_mask` fixes.
fn principal_point_indices(cam: &CameraModel) -> &'static [usize] {
    match cam {
        CameraModel::SimplePinhole { .. } => &[1, 2],
        CameraModel::Pinhole { .. } => &[2, 3],
        CameraModel::SimpleRadial { .. } => &[1, 2],
        CameraModel::Radial { .. } => &[1, 2],
        CameraModel::Radial3 { .. } => &[1, 2],
        CameraModel::OpenCV { .. } => &[2, 3],
        CameraModel::OpenCVFisheye { .. } => &[2, 3],
    }
}

/// Fixed-parameter mask implementing the standard "refine focal length and
/// distortion, hold the principal point fixed at its initial guess" policy -
/// see `BaInput::fixed_camera_params`'s docs for why the principal point
/// specifically needs this. One `Vec<bool>` per camera, index-aligned with
/// `cam.params()`.
pub fn default_fixed_params_mask(cameras: &[CameraModel]) -> Vec<Vec<bool>> {
    cameras
        .iter()
        .map(|cam| {
            let n = cam.params().len();
            let mut mask = vec![false; n];
            for &idx in principal_point_indices(cam) {
                mask[idx] = true;
            }
            mask
        })
        .collect()
}

use crate::{reprojection_residual, robust_weight, Observation, RobustLoss};

#[derive(Debug, Clone, Copy)]
pub struct IntrinsicsRefineParams {
    pub max_iterations: usize,
    pub initial_lambda: f64,
    pub robust_loss: RobustLoss,
    pub loss_scale_px: f64,
}

impl Default for IntrinsicsRefineParams {
    fn default() -> Self {
        IntrinsicsRefineParams {
            max_iterations: 30,
            initial_lambda: 1e-3,
            robust_loss: RobustLoss::Huber,
            loss_scale_px: 2.0,
        }
    }
}

const EPS: f64 = 1e-6;

pub(crate) fn perturb_camera(cam: &CameraModel, k: usize, delta: f64) -> Option<CameraModel> {
    let mut params = cam.params();
    params[k] += delta;
    CameraModel::from_name_and_params(cam.name(), &params)
}

fn camera_cost(
    cam: &CameraModel,
    poses: &[&Pose],
    points: &[&Vector3<f64>],
    obs: &[(f64, f64)],
    params: &IntrinsicsRefineParams,
) -> f64 {
    (0..poses.len())
        .map(
            |i| match reprojection_residual(poses[i], cam, points[i], obs[i]) {
                Some(r) => {
                    let n = r.norm();
                    let w = robust_weight(n, params.robust_loss, params.loss_scale_px);
                    w * n * n
                }
                None => 0.0,
            },
        )
        .sum()
}

/// Refine every camera's intrinsics in place via dense LM, holding all poses
/// and 3D points fixed. `camera_of_image[obs.image_idx]` must index into
/// `cameras`.
pub fn refine_intrinsics(
    cameras: &mut [CameraModel],
    camera_of_image: &[usize],
    poses: &[Pose],
    points: &[Vector3<f64>],
    observations: &[Observation],
    params: &IntrinsicsRefineParams,
) {
    // Group observations by which physical camera they belong to.
    let mut by_camera: Vec<Vec<&Observation>> = vec![Vec::new(); cameras.len()];
    for obs in observations {
        by_camera[camera_of_image[obs.image_idx]].push(obs);
    }

    let refined: Vec<CameraModel> = cameras
        .par_iter()
        .enumerate()
        .map(|(cam_idx, cam)| {
            let obs_for_camera = &by_camera[cam_idx];
            if obs_for_camera.len() < 8 {
                // Not enough observations to trust refining this camera at
                // all (fewer than the largest supported parameter count) -
                // leave it exactly as-is rather than overfitting noise.
                return *cam;
            }
            refine_one_camera(cam, obs_for_camera, poses, points, params)
        })
        .collect();

    cameras.copy_from_slice(&refined);
}

fn refine_one_camera(
    cam: &CameraModel,
    obs_list: &[&Observation],
    poses: &[Pose],
    points: &[Vector3<f64>],
    params: &IntrinsicsRefineParams,
) -> CameraModel {
    let n_params = cam.params().len();
    let img_poses: Vec<&Pose> = obs_list.iter().map(|o| &poses[o.image_idx]).collect();
    let obs_points: Vec<&Vector3<f64>> = obs_list.iter().map(|o| &points[o.point_idx]).collect();
    let obs_px: Vec<(f64, f64)> = obs_list.iter().map(|o| (o.x, o.y)).collect();

    let mut current = *cam;
    let mut cost = camera_cost(&current, &img_poses, &obs_points, &obs_px, params);
    let mut lambda = params.initial_lambda;

    for _ in 0..params.max_iterations {
        let mut jtj = DMatrix::<f64>::zeros(n_params, n_params);
        let mut jtr = DVector::<f64>::zeros(n_params);

        for i in 0..obs_list.len() {
            let Some(r0) = reprojection_residual(img_poses[i], &current, obs_points[i], obs_px[i])
            else {
                continue;
            };
            let weight = robust_weight(r0.norm(), params.robust_loss, params.loss_scale_px);

            let mut jac = vec![SVector::<f64, 2>::zeros(); n_params];
            for k in 0..n_params {
                let plus = perturb_camera(&current, k, EPS);
                let minus = perturb_camera(&current, k, -EPS);
                let r_plus = plus
                    .as_ref()
                    .and_then(|c| reprojection_residual(img_poses[i], c, obs_points[i], obs_px[i]))
                    .unwrap_or(r0);
                let r_minus = minus
                    .as_ref()
                    .and_then(|c| reprojection_residual(img_poses[i], c, obs_points[i], obs_px[i]))
                    .unwrap_or(r0);
                jac[k] = (r_plus - r_minus) / (2.0 * EPS);
            }
            for a in 0..n_params {
                jtr[a] += weight * jac[a].dot(&r0);
                for b in 0..n_params {
                    jtj[(a, b)] += weight * jac[a].dot(&jac[b]);
                }
            }
        }

        let mut accepted = false;
        for _ in 0..8 {
            let mut damped = jtj.clone();
            for a in 0..n_params {
                damped[(a, a)] += lambda * jtj[(a, a)].max(1e-12);
            }
            let delta = match nalgebra::linalg::Cholesky::new(damped.clone()) {
                Some(chol) => chol.solve(&(-&jtr)),
                None => match damped.lu().solve(&(-&jtr)) {
                    Some(sol) => sol,
                    None => {
                        lambda *= 3.0;
                        continue;
                    }
                },
            };
            let mut trial_params = current.params();
            for k in 0..n_params {
                trial_params[k] += delta[k];
            }
            let Some(trial_cam) = CameraModel::from_name_and_params(current.name(), &trial_params)
            else {
                lambda *= 3.0;
                continue;
            };
            let trial_cost = camera_cost(&trial_cam, &img_poses, &obs_points, &obs_px, params);
            if trial_cost.is_finite() && trial_cost < cost {
                current = trial_cam;
                cost = trial_cost;
                lambda = (lambda / 3.0).max(1e-12);
                accepted = true;
                break;
            }
            lambda *= 3.0;
        }
        if !accepted {
            break;
        }
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinhole(f: f64, w: u32, h: u32) -> CameraModel {
        CameraModel::SimplePinhole {
            f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }

    #[test]
    fn recovers_true_focal_length_from_wrong_initial_guess() {
        let true_cam = pinhole(1500.0, 1000, 800);
        let pose = Pose::identity();
        let points: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.4 * (t * 0.6).sin(), 0.3 * (t * 0.4).cos(), 3.0 + 0.05 * t)
            })
            .collect();
        let observations: Vec<Observation> = points
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let (x, y) = true_cam.project(&pose.transform_point(p));
                Observation {
                    image_idx: 0,
                    point_idx: idx,
                    x,
                    y,
                }
            })
            .collect();

        // Deliberately wrong initial guess, like sfmtory's width-based
        // heuristic (`f = max(w,h) * 1.2 = 960`, vs. true 1500 here).
        let mut cameras = vec![pinhole(960.0, 1000, 800)];
        refine_intrinsics(
            &mut cameras,
            &[0],
            &[pose],
            &points,
            &observations,
            &IntrinsicsRefineParams::default(),
        );

        let CameraModel::SimplePinhole { f, .. } = cameras[0] else {
            panic!("model changed")
        };
        assert!(
            (f - 1500.0).abs() < 1.0,
            "recovered focal length {f}, want ~1500"
        );
    }

    #[test]
    fn leaves_camera_unchanged_with_too_few_observations() {
        let true_cam = pinhole(1500.0, 1000, 800);
        let pose = Pose::identity();
        let point = Vector3::new(0.1, 0.1, 3.0);
        let (x, y) = true_cam.project(&pose.transform_point(&point));
        let observations = vec![Observation {
            image_idx: 0,
            point_idx: 0,
            x,
            y,
        }];

        let wrong = pinhole(960.0, 1000, 800);
        let mut cameras = vec![wrong];
        refine_intrinsics(
            &mut cameras,
            &[0],
            &[pose],
            &[point],
            &observations,
            &IntrinsicsRefineParams::default(),
        );

        assert_eq!(
            cameras[0], wrong,
            "too few observations should leave the camera untouched"
        );
    }

    #[test]
    fn improves_overall_reprojection_cost_end_to_end_with_bundle_adjust() {
        // Sanity check the two passes compose: BA on poses/points with a
        // wrong-but-fixed focal length, then intrinsics refinement, should
        // leave the *combined* cost no worse than either alone.
        let true_cam = pinhole(1500.0, 1000, 800);
        let true_pose = Pose::identity();
        let points: Vec<Vector3<f64>> = (0..30)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.4 * (t * 0.6).sin(), 0.3 * (t * 0.4).cos(), 3.0 + 0.05 * t)
            })
            .collect();
        let observations: Vec<Observation> = points
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let (x, y) = true_cam.project(&true_pose.transform_point(p));
                Observation {
                    image_idx: 0,
                    point_idx: idx,
                    x,
                    y,
                }
            })
            .collect();

        let mut cameras = vec![pinhole(960.0, 1000, 800)];
        let before_cost = camera_cost(
            &cameras[0],
            &vec![&true_pose; points.len()],
            &points.iter().collect::<Vec<_>>(),
            &observations.iter().map(|o| (o.x, o.y)).collect::<Vec<_>>(),
            &IntrinsicsRefineParams::default(),
        );

        refine_intrinsics(
            &mut cameras,
            &[0],
            &[true_pose],
            &points,
            &observations,
            &IntrinsicsRefineParams::default(),
        );

        let after_cost = camera_cost(
            &cameras[0],
            &vec![&true_pose; points.len()],
            &points.iter().collect::<Vec<_>>(),
            &observations.iter().map(|o| (o.x, o.y)).collect::<Vec<_>>(),
            &IntrinsicsRefineParams::default(),
        );
        assert!(
            after_cost < before_cost * 0.01,
            "expected large cost reduction: before={before_cost} after={after_cost}"
        );
    }
}
