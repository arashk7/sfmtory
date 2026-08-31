//! Choosing a camera model by how well it *generalises*, not how well it fits.
//!
//! A richer camera model always fits the observations it was fitted to at
//! least as well as a simpler one, so in-sample reprojection error cannot
//! choose between them - it just ranks them by parameter count. The usual
//! remedies are an information criterion (AIC/BIC) or cross-validation.
//!
//! Cross-validation is used here, and the fold structure is the part that
//! matters. Observations inside one capture share a board pose, so they are
//! correlated; splitting them at random lets a model fitted on some of a
//! capture's points be scored on the rest of the *same* capture, which it can
//! predict partly by having absorbed that pose's particular noise. That is
//! optimistic in exactly the direction that over-selects complex models. AIC
//! and BIC have the same problem for a different reason - both count
//! observations as independent, which these are not.
//!
//! So folds are whole captures where the dataset has them (a rig photographing
//! a target that moved between sessions gives genuinely independent groups),
//! and whole images otherwise. Fit on the other folds, score on the held-out
//! one, and a parameter that only ever memorised one board pose scores badly.
//!
//! Poses and 3D points are held fixed at the reconstruction's values
//! throughout, so this measures the camera model alone. That is not perfectly
//! clean - the structure was itself estimated using every capture under the
//! original model, so a little of the held-out fold's information reaches the
//! fit through the points. Re-running the whole reconstruction per fold would
//! remove that and cost orders of magnitude more; the residual leakage is the
//! same for every candidate, so it biases the *absolute* numbers rather than
//! the ranking between them.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use nalgebra::Vector3;
use sfm_ba::{refine_intrinsics, IntrinsicsRefineParams, Observation};
use sfm_core::{CameraModel, Reconstruction};

/// One candidate model's score for one physical camera.
#[derive(Debug, Clone)]
pub struct ModelScore {
    pub name: &'static str,
    pub num_params: usize,
    /// Mean reprojection error on folds the intrinsics were not fitted to.
    pub held_out_px: f64,
    /// Mean reprojection error on the fitting folds, for comparison. A model
    /// whose in-sample error keeps falling while held-out error rises is
    /// overfitting, and showing both makes that visible.
    pub in_sample_px: f64,
    /// Folds that produced a usable fit.
    pub folds: usize,
}

#[derive(Debug, Clone)]
pub struct CameraChoice {
    pub camera_id: u32,
    pub num_observations: usize,
    pub scores: Vec<ModelScore>,
    /// Best held-out error, tie-broken toward fewer parameters.
    pub recommended: &'static str,
}

/// How much better a richer model must be before it is worth its parameters.
///
/// Held-out error already penalises overfitting, but it is itself a noisy
/// estimate from a handful of folds, so a coin-flip difference should not
/// promote an 8-parameter model over a 4-parameter one. 2% is small enough to
/// let a genuinely better model win and large enough that noise does not.
const IMPROVEMENT_MARGIN: f64 = 0.02;

/// Candidate models, simplest first. `SIMPLE_PINHOLE` is omitted: a model with
/// no distortion term at all is never the right answer for a real lens, and
/// including it only invites it to win on a camera with too few observations
/// to fit anything.
pub const CANDIDATES: [&str; 5] = [
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
];

/// Which fold each observation belongs to.
pub struct Folds {
    /// Fold index per observation, parallel to the observation list.
    pub of_observation: Vec<usize>,
    pub count: usize,
    /// What the folds are, for reporting.
    pub kind: &'static str,
}

/// Builds the observation list the fitter wants, plus a per-camera index.
pub struct Problem {
    pub observations: Vec<Observation>,
    pub poses: Vec<sfm_core::Pose>,
    pub points: Vec<Vector3<f64>>,
    pub cameras: Vec<CameraModel>,
    pub camera_ids: Vec<u32>,
    pub camera_of_image: Vec<usize>,
    /// Image ids, index-aligned with `poses`.
    pub image_ids: Vec<u32>,
}

impl Problem {
    pub fn from_reconstruction(recon: &Reconstruction) -> Result<Self> {
        let camera_ids: Vec<u32> = recon.cameras.keys().copied().collect();
        let cameras: Vec<CameraModel> = recon.cameras.values().map(|c| c.model).collect();
        let camera_slot: BTreeMap<u32, usize> = camera_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        let image_ids: Vec<u32> = recon.images.keys().copied().collect();
        let image_slot: BTreeMap<u32, usize> = image_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let poses: Vec<sfm_core::Pose> = recon.images.values().map(|im| im.pose).collect();
        let camera_of_image: Vec<usize> = recon
            .images
            .values()
            .map(|im| camera_slot[&im.camera_id])
            .collect();

        let point_ids: Vec<u64> = recon.points3d.keys().copied().collect();
        let point_slot: BTreeMap<u64, usize> = point_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let points: Vec<Vector3<f64>> = recon.points3d.values().map(|p| p.xyz).collect();

        let mut observations = Vec::new();
        for point in recon.points3d.values() {
            for t in &point.track {
                let (Some(&image_idx), Some(&point_idx)) =
                    (image_slot.get(&t.image_id), point_slot.get(&point.id))
                else {
                    continue;
                };
                let Some(image) = recon.images.get(&t.image_id) else {
                    continue;
                };
                let Some(&(x, y)) = image.keypoints.get(t.point2d_idx as usize) else {
                    continue;
                };
                observations.push(Observation {
                    image_idx,
                    point_idx,
                    x: x as f64,
                    y: y as f64,
                });
            }
        }
        if observations.is_empty() {
            bail!("the reconstruction has no observations to fit a camera model to");
        }
        Ok(Problem {
            observations,
            poses,
            points,
            cameras,
            camera_ids,
            camera_of_image,
            image_ids,
        })
    }

    /// Mean reprojection error over the given observations for one camera.
    fn error_on(&self, cameras: &[CameraModel], which: &[usize]) -> (f64, usize) {
        let mut sum = 0.0;
        let mut n = 0usize;
        for &i in which {
            let o = &self.observations[i];
            let cam = &cameras[self.camera_of_image[o.image_idx]];
            let pc = self.poses[o.image_idx].transform_point(&self.points[o.point_idx]);
            if pc.z <= 1e-9 {
                continue;
            }
            let (px, py) = cam.project(&pc);
            sum += ((px - o.x).powi(2) + (py - o.y).powi(2)).sqrt();
            n += 1;
        }
        (if n > 0 { sum / n as f64 } else { f64::NAN }, n)
    }
}

/// Re-seeds a camera as `name`, carrying the focal length and principal point
/// across and starting every distortion term at zero.
///
/// Starting each candidate from the same geometry rather than from its own
/// guess keeps the comparison about the model rather than about who got the
/// luckier initialisation.
pub fn reseed(cam: &CameraModel, name: &str) -> Option<CameraModel> {
    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    let params: Vec<f64> = match name {
        "SIMPLE_PINHOLE" => vec![fx, cx, cy],
        "PINHOLE" => vec![fx, fy, cx, cy],
        "SIMPLE_RADIAL" => vec![fx, cx, cy, 0.0],
        "RADIAL" => vec![fx, cx, cy, 0.0, 0.0],
        "OPENCV" | "OPENCV_FISHEYE" => vec![fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0],
        _ => return None,
    };
    CameraModel::from_name_and_params(name, &params)
}

/// Scores every candidate model on every camera by held-out reprojection error.
pub fn select(problem: &Problem, folds: &Folds) -> Vec<CameraChoice> {
    let num_cameras = problem.cameras.len();
    // [camera][model] -> accumulated held-out and in-sample error.
    let mut held: Vec<Vec<(f64, f64, usize)>> =
        vec![vec![(0.0, 0.0, 0); CANDIDATES.len()]; num_cameras];

    for (m, name) in CANDIDATES.iter().enumerate() {
        for fold in 0..folds.count {
            let (train, test): (Vec<usize>, Vec<usize>) =
                (0..problem.observations.len()).partition(|&i| folds.of_observation[i] != fold);
            if train.is_empty() || test.is_empty() {
                continue;
            }
            let Some(seeded): Option<Vec<CameraModel>> = problem
                .cameras
                .iter()
                .map(|c| reseed(c, name))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let mut fitted = seeded;
            let train_obs: Vec<Observation> =
                train.iter().map(|&i| problem.observations[i]).collect();
            refine_intrinsics(
                &mut fitted,
                &problem.camera_of_image,
                &problem.poses,
                &problem.points,
                &train_obs,
                &IntrinsicsRefineParams::default(),
            );

            // Score per camera, so a model can win on one lens and lose on
            // another - which is the whole point of a per-camera choice.
            for (cam_idx, acc) in held.iter_mut().enumerate() {
                let mine = |set: &[usize]| -> Vec<usize> {
                    set.iter()
                        .copied()
                        .filter(|&i| {
                            problem.camera_of_image[problem.observations[i].image_idx] == cam_idx
                        })
                        .collect()
                };
                let (test_err, test_n) = problem.error_on(&fitted, &mine(&test));
                let (train_err, _) = problem.error_on(&fitted, &mine(&train));
                if test_n == 0 || !test_err.is_finite() || !train_err.is_finite() {
                    continue;
                }
                let slot = &mut acc[m];
                slot.0 += test_err;
                slot.1 += train_err;
                slot.2 += 1;
            }
        }
    }

    (0..num_cameras)
        .map(|cam_idx| {
            let scores: Vec<ModelScore> = CANDIDATES
                .iter()
                .enumerate()
                .filter(|(m, _)| held[cam_idx][*m].2 > 0)
                .map(|(m, name)| {
                    let (sum_test, sum_train, k) = held[cam_idx][m];
                    ModelScore {
                        name,
                        num_params: reseed(&problem.cameras[cam_idx], name)
                            .map(|c| c.params().len())
                            .unwrap_or(0),
                        held_out_px: sum_test / k as f64,
                        in_sample_px: sum_train / k as f64,
                        folds: k,
                    }
                })
                .collect();
            // Simplest model within the margin of the best, so extra
            // parameters have to earn their place.
            let best = scores
                .iter()
                .map(|s| s.held_out_px)
                .fold(f64::MAX, f64::min);
            let recommended = scores
                .iter()
                .find(|s| s.held_out_px <= best * (1.0 + IMPROVEMENT_MARGIN))
                .map(|s| s.name)
                .unwrap_or("SIMPLE_RADIAL");
            let num_observations = problem
                .observations
                .iter()
                .filter(|o| problem.camera_of_image[o.image_idx] == cam_idx)
                .count();
            CameraChoice {
                camera_id: problem.camera_ids[cam_idx],
                num_observations,
                scores,
                recommended,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfm_core::{Camera, Image, Point3D, Pose, TrackElement};

    /// Builds a reconstruction whose images observe a grid of points through
    /// `truth`, so a model that can represent `truth` should win.
    fn synthetic(truth: CameraModel, jitter: f64) -> Reconstruction {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: CameraModel::SimpleRadial {
                    f: 900.0,
                    cx: 640.0,
                    cy: 512.0,
                    k: 0.0,
                },
                width: 1280,
                height: 1024,
            },
        );
        let mut pid = 0u64;
        let mut points = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                pid += 1;
                let p = Vector3::new(
                    (i as f64 - 5.5) * 0.14,
                    (j as f64 - 5.5) * 0.14,
                    3.0 + ((i + j) % 3) as f64 * 0.05,
                );
                points.push((pid, p));
            }
        }
        // Four "captures", each a different view of the same points.
        for (img_id, (dx, dz)) in [(0.0, 0.0), (0.5, 0.2), (-0.4, 0.1), (0.2, -0.3)]
            .into_iter()
            .enumerate()
        {
            let id = img_id as u32 + 1;
            let pose = Pose {
                rotation: nalgebra::UnitQuaternion::identity(),
                translation: Vector3::new(dx, 0.1 * id as f64, dz),
            };
            let mut keypoints = Vec::new();
            let mut point3d_ids = Vec::new();
            for (k, (pid, p)) in points.iter().enumerate() {
                let (x, y) = truth.project(&pose.transform_point(p));
                // Deterministic pseudo-noise, so the test is repeatable.
                let n = ((k as f64 * 12.9898 + id as f64 * 78.233).sin() * 43758.5453).fract();
                keypoints.push(((x + n * jitter) as f32, (y - n * jitter) as f32));
                point3d_ids.push(Some(*pid));
            }
            recon.images.insert(
                id,
                Image {
                    id,
                    camera_id: 1,
                    name: format!("{id}.jpg"),
                    pose,
                    keypoints,
                    point3d_ids,
                },
            );
        }
        for (k, (pid, p)) in points.iter().enumerate() {
            recon.points3d.insert(
                *pid,
                Point3D {
                    id: *pid,
                    xyz: *p,
                    color: [0, 0, 0],
                    error: 0.0,
                    track: (1..=4u32)
                        .map(|image_id| TrackElement {
                            image_id,
                            point2d_idx: k as u32,
                        })
                        .collect(),
                },
            );
        }
        recon
    }

    fn per_image_folds(p: &Problem) -> Folds {
        Folds {
            of_observation: p.observations.iter().map(|o| o.image_idx).collect(),
            count: p.poses.len(),
            kind: "image",
        }
    }

    #[test]
    fn reseeding_carries_geometry_and_zeroes_distortion() {
        let cam = CameraModel::SimpleRadial {
            f: 1234.0,
            cx: 640.0,
            cy: 512.0,
            k: -0.3,
        };
        let opencv = reseed(&cam, "OPENCV").unwrap();
        assert_eq!(opencv.name(), "OPENCV");
        assert_eq!(opencv.focal_lengths(), (1234.0, 1234.0));
        assert_eq!(opencv.principal_point(), (640.0, 512.0));
        // The old k must not leak into the new model's distortion terms.
        assert!(opencv.opencv_distortion().iter().all(|d| *d == 0.0));
        assert!(reseed(&cam, "NOT_A_MODEL").is_none());
    }

    /// A lens with real radial distortion: the model that can represent it has
    /// to beat the one that cannot.
    #[test]
    fn a_distorted_lens_prefers_a_model_with_distortion() {
        let truth = CameraModel::SimpleRadial {
            f: 900.0,
            cx: 640.0,
            cy: 512.0,
            k: -0.18,
        };
        let recon = synthetic(truth, 0.0);
        let problem = Problem::from_reconstruction(&recon).unwrap();
        let folds = per_image_folds(&problem);
        let choices = select(&problem, &folds);
        assert_eq!(choices.len(), 1);
        let by_name = |n: &str| {
            choices[0]
                .scores
                .iter()
                .find(|s| s.name == n)
                .unwrap()
                .held_out_px
        };
        assert!(
            by_name("SIMPLE_RADIAL") < by_name("PINHOLE") * 0.5,
            "distortion-capable model should be much better: {:?}",
            choices[0].scores
        );
        assert_ne!(choices[0].recommended, "PINHOLE");
    }

    /// The case the margin exists for: with no distortion to find, the extra
    /// parameters must not win.
    #[test]
    fn an_undistorted_lens_is_not_given_spurious_parameters() {
        let truth = CameraModel::Pinhole {
            fx: 900.0,
            fy: 900.0,
            cx: 640.0,
            cy: 512.0,
        };
        let recon = synthetic(truth, 0.4);
        let problem = Problem::from_reconstruction(&recon).unwrap();
        let folds = per_image_folds(&problem);
        let choices = select(&problem, &folds);
        let rec = choices[0].recommended;
        assert!(
            rec == "PINHOLE" || rec == "SIMPLE_RADIAL",
            "expected a simple model on an undistorted lens, got {rec}"
        );
        assert!(choices[0].num_observations > 0);
    }

    #[test]
    fn every_candidate_is_scored_on_every_fold_it_can_fit() {
        let truth = CameraModel::SimpleRadial {
            f: 900.0,
            cx: 640.0,
            cy: 512.0,
            k: -0.1,
        };
        let recon = synthetic(truth, 0.1);
        let problem = Problem::from_reconstruction(&recon).unwrap();
        let folds = per_image_folds(&problem);
        assert_eq!(folds.count, 4);
        let choices = select(&problem, &folds);
        for s in &choices[0].scores {
            assert_eq!(s.folds, 4, "{} was not scored on every fold", s.name);
            assert!(s.held_out_px.is_finite());
            assert!(s.in_sample_px.is_finite());
        }
    }
}
