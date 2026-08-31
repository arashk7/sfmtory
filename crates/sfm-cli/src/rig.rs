//! Calibrating a rig of fixed cameras from several captures of a moved target.
//!
//! The ordinary pipeline treats a project as one scene. A scanner rig is not
//! one scene: the cameras never move, the target does, and each capture is a
//! *separate* scene that happens to be observed by the same hardware. Marker
//! identities are stamped per capture precisely so a moved marker cannot match
//! itself across them, which leaves the captures as disconnected components of
//! the match graph - and an incremental reconstruction can only grow one of
//! them.
//!
//! `--merge-multicaps` works around that by pooling every capture's corners
//! into one image per camera. It buys observations for intrinsics and it costs
//! the geometry dearly, because the pooled corners are *different 3D points*
//! per capture: a point can then be seen by at most as many cameras as saw
//! that one marker, and multi-view tracks collapse. Measured on a 4-camera
//! wall rig: merged, the recovered centres sat 16.2% of their mean spacing off
//! a common plane with a longest track of 3; the same cameras from a single
//! capture sat **1.1% off-plane** with 60 points seen by all four.
//!
//! So this reconstructs each capture on its own, where tracks are long and the
//! geometry is well conditioned, and then exploits the thing that makes it a
//! rig: every capture is an independent estimate of the *same* camera
//! arrangement. Aligning them by their shared cameras and comparing gives both
//! a better estimate and something no single reconstruction can offer - a
//! measured spread per camera, which is a direct statement of how well the
//! calibration is actually determined.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use nalgebra::{Matrix3, Vector3};
use sfm_core::Reconstruction;

/// One capture's view of the rig: where each camera ended up.
pub struct CaptureRig {
    pub capture_id: i64,
    pub centres: BTreeMap<u32, Vector3<f64>>,
    /// Camera-to-world rotation per camera.
    pub rotations: BTreeMap<u32, Matrix3<f64>>,
    pub num_points: usize,
    pub mean_reprojection_px: f64,
}

impl CaptureRig {
    pub fn from_reconstruction(capture_id: i64, recon: &Reconstruction) -> Self {
        let mut centres = BTreeMap::new();
        let mut rotations = BTreeMap::new();
        for im in recon.images.values() {
            centres.insert(im.camera_id, im.pose.camera_center());
            rotations.insert(
                im.camera_id,
                im.pose
                    .rotation
                    .to_rotation_matrix()
                    .into_inner()
                    .transpose(),
            );
        }
        CaptureRig {
            capture_id,
            centres,
            rotations,
            num_points: recon.points3d.len(),
            mean_reprojection_px: recon.mean_reprojection_error(),
        }
    }
}

/// The similarity transform taking `from` onto `to`, in the least-squares
/// sense (Umeyama).
///
/// A similarity, not a rigid motion: each capture is reconstructed
/// independently and monocular structure-from-motion fixes nothing about
/// absolute scale, so two captures of the same rig legitimately come out at
/// different sizes. Solving for scale is what lets them be compared at all.
pub struct Similarity {
    pub scale: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl Similarity {
    pub fn apply(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.scale * (self.rotation * p) + self.translation
    }
}

pub fn umeyama(from: &[Vector3<f64>], to: &[Vector3<f64>]) -> Option<Similarity> {
    let n = from.len();
    if n < 3 || n != to.len() {
        return None;
    }
    let nf = n as f64;
    let mu_f: Vector3<f64> = from.iter().sum::<Vector3<f64>>() / nf;
    let mu_t: Vector3<f64> = to.iter().sum::<Vector3<f64>>() / nf;
    let mut cov = Matrix3::zeros();
    let mut var_f = 0.0;
    for (f, t) in from.iter().zip(to) {
        let df = f - mu_f;
        let dt = t - mu_t;
        cov += dt * df.transpose();
        var_f += df.norm_squared();
    }
    cov /= nf;
    var_f /= nf;
    if var_f < 1e-12 {
        return None;
    }
    let svd = cov.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    // Guard against a reflection: the SVD is happy to return one, and a
    // mirrored rig would align beautifully while being physically impossible.
    let mut s = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let rotation = u * s * v_t;
    let d = svd.singular_values;
    let trace = d[0] * s[(0, 0)] + d[1] * s[(1, 1)] + d[2] * s[(2, 2)];
    let scale = trace / var_f;
    Some(Similarity {
        scale,
        rotation,
        translation: mu_t - scale * (rotation * mu_f),
    })
}

/// One camera's position across every capture that saw it.
pub struct CameraSpread {
    pub camera_id: u32,
    pub mean: Vector3<f64>,
    /// RMS distance of the per-capture estimates from their mean, in the
    /// reference capture's units.
    pub rms: f64,
    /// Largest angle between this camera's optical axis in any two captures,
    /// in degrees. Position can agree while orientation does not, and for a
    /// rig the orientation is half the calibration.
    pub axis_spread_deg: f64,
    pub observations: usize,
}

pub struct RigSolution {
    pub reference: i64,
    pub cameras: Vec<CameraSpread>,
    /// Captures that could not be aligned, with why.
    pub skipped: Vec<(i64, String)>,
    /// Mean spacing between cameras, the scale everything else is relative to.
    pub mean_spacing: f64,
}

/// Brings every capture into the reference capture's frame and reports where
/// each camera sits, with its spread.
pub fn solve(rigs: &[CaptureRig], min_shared: usize) -> Result<RigSolution> {
    let Some(reference) = rigs.iter().max_by_key(|r| r.centres.len()) else {
        bail!("no capture produced a reconstruction");
    };
    if reference.centres.len() < 3 {
        bail!(
            "the best capture registered only {} camera(s); at least 3 are needed to align \
             captures to each other",
            reference.centres.len()
        );
    }

    let mut accum: BTreeMap<u32, Vec<Vector3<f64>>> = BTreeMap::new();
    let mut axes: BTreeMap<u32, Vec<Vector3<f64>>> = BTreeMap::new();
    let mut skipped = Vec::new();
    for rig in rigs {
        if rig.capture_id == reference.capture_id {
            for (id, c) in &rig.centres {
                accum.entry(*id).or_default().push(*c);
                if let Some(r) = rig.rotations.get(id) {
                    axes.entry(*id).or_default().push(r * Vector3::z());
                }
            }
            continue;
        }
        let shared: Vec<u32> = rig
            .centres
            .keys()
            .filter(|id| reference.centres.contains_key(id))
            .copied()
            .collect();
        if shared.len() < min_shared.max(3) {
            skipped.push((
                rig.capture_id,
                format!(
                    "shares only {} camera(s) with the reference; needs {}",
                    shared.len(),
                    min_shared.max(3)
                ),
            ));
            continue;
        }
        let from: Vec<Vector3<f64>> = shared.iter().map(|id| rig.centres[id]).collect();
        let to: Vec<Vector3<f64>> = shared.iter().map(|id| reference.centres[id]).collect();
        let Some(sim) = umeyama(&from, &to) else {
            skipped.push((rig.capture_id, "cameras are degenerate (collinear?)".into()));
            continue;
        };
        for (id, c) in &rig.centres {
            accum.entry(*id).or_default().push(sim.apply(c));
            if let Some(r) = rig.rotations.get(id) {
                // Direction only, so the similarity's scale and translation
                // drop out and just the rotation applies.
                axes.entry(*id)
                    .or_default()
                    .push(sim.rotation * (r * Vector3::z()));
            }
        }
    }

    let mut cameras = Vec::new();
    for (camera_id, positions) in accum {
        let n = positions.len() as f64;
        let mean: Vector3<f64> = positions.iter().sum::<Vector3<f64>>() / n;
        let rms = (positions
            .iter()
            .map(|p| (p - mean).norm_squared())
            .sum::<f64>()
            / n)
            .sqrt();
        let axis_spread_deg = axes
            .get(&camera_id)
            .map(|dirs| {
                let mut worst: f64 = 0.0;
                for i in 0..dirs.len() {
                    for j in (i + 1)..dirs.len() {
                        let c = dirs[i].dot(&dirs[j]).clamp(-1.0, 1.0);
                        worst = worst.max(c.acos().to_degrees());
                    }
                }
                worst
            })
            .unwrap_or(0.0);
        cameras.push(CameraSpread {
            camera_id,
            mean,
            rms,
            axis_spread_deg,
            observations: positions.len(),
        });
    }

    let mut spacings = Vec::new();
    for i in 0..cameras.len() {
        for j in (i + 1)..cameras.len() {
            spacings.push((cameras[i].mean - cameras[j].mean).norm());
        }
    }
    let mean_spacing = if spacings.is_empty() {
        1.0
    } else {
        spacings.iter().sum::<f64>() / spacings.len() as f64
    };

    Ok(RigSolution {
        reference: reference.capture_id,
        cameras,
        skipped,
        mean_spacing,
    })
}

/// How far the cameras sit from their own best-fit plane, as a fraction of
/// their mean spacing.
///
/// A rig whose cameras are mounted on one wall should measure near zero, and
/// this is the cheapest independent check that a calibration is right: it uses
/// a fact about the hardware that no part of the reconstruction was told.
pub fn planarity(points: &[Vector3<f64>]) -> Option<(f64, f64)> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f64;
    let mu: Vector3<f64> = points.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    for p in points {
        let d = p - mu;
        cov += d * d.transpose();
    }
    cov /= n;
    let eig = cov.symmetric_eigen();
    let mut ev: Vec<f64> = eig.eigenvalues.iter().map(|v| v.max(0.0).sqrt()).collect();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut spacings = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            spacings.push((points[i] - points[j]).norm());
        }
    }
    let mean_spacing = spacings.iter().sum::<f64>() / spacings.len() as f64;
    if mean_spacing < 1e-12 {
        return None;
    }
    // (out-of-plane extent / mean spacing, out-of-plane / in-plane extent)
    Some((ev[0] / mean_spacing, ev[0] / ev[2].max(1e-12)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn umeyama_recovers_a_known_similarity() {
        let from = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let angle = 0.7_f64;
        let rot = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let (scale, t) = (2.5, v(3.0, -1.0, 0.5));
        let to: Vec<Vector3<f64>> = from.iter().map(|p| scale * (rot * p) + t).collect();
        let sim = umeyama(&from, &to).expect("well-conditioned");
        assert!((sim.scale - scale).abs() < 1e-9, "scale {}", sim.scale);
        for (f, expected) in from.iter().zip(&to) {
            assert!((sim.apply(f) - expected).norm() < 1e-9);
        }
    }

    /// The SVD will happily hand back a reflection, which would align a
    /// mirrored rig perfectly while being physically impossible.
    #[test]
    fn umeyama_never_returns_a_reflection() {
        let from = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let mirrored: Vec<Vector3<f64>> = from.iter().map(|p| v(p.x, p.y, -p.z)).collect();
        let sim = umeyama(&from, &mirrored).unwrap();
        assert!(
            sim.rotation.determinant() > 0.0,
            "returned a reflection: det {}",
            sim.rotation.determinant()
        );
    }

    #[test]
    fn umeyama_refuses_degenerate_input() {
        assert!(umeyama(
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)]
        )
        .is_none());
        let same = vec![v(1.0, 1.0, 1.0); 4];
        assert!(
            umeyama(&same, &same).is_none(),
            "zero variance must not divide"
        );
    }

    #[test]
    fn planarity_is_zero_on_a_wall_and_large_on_a_cube() {
        let wall = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(1.0, 1.0, 0.0),
        ];
        let (off, ratio) = planarity(&wall).unwrap();
        assert!(off < 1e-9, "a wall is planar, got {off}");
        assert!(ratio < 1e-9);

        let cube = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let (off_c, _) = planarity(&cube).unwrap();
        assert!(off_c > 0.1, "a tetrahedron is not planar, got {off_c}");
    }

    /// Independent captures of one rig must agree once aligned - that is the
    /// whole premise, so it is pinned.
    #[test]
    fn independently_scaled_captures_agree_after_alignment() {
        let truth: BTreeMap<u32, Vector3<f64>> = [
            (1, v(0.0, 0.0, 0.0)),
            (2, v(1.0, 0.0, 0.0)),
            (3, v(1.0, 1.0, 0.0)),
            (4, v(0.0, 1.0, 0.0)),
        ]
        .into_iter()
        .collect();
        let rigs: Vec<CaptureRig> = [1.0, 2.0, 0.5]
            .iter()
            .enumerate()
            .map(|(i, s)| CaptureRig {
                capture_id: i as i64,
                centres: truth.iter().map(|(k, p)| (*k, *s * p)).collect(),
                rotations: BTreeMap::new(),
                num_points: 100,
                mean_reprojection_px: 0.5,
            })
            .collect();
        let sol = solve(&rigs, 3).unwrap();
        assert!(sol.skipped.is_empty());
        for c in &sol.cameras {
            assert_eq!(c.observations, 3);
            assert_eq!(c.axis_spread_deg, 0.0);
            assert!(
                c.rms < 1e-9,
                "camera {} disagreed by {} after alignment",
                c.camera_id,
                c.rms
            );
        }
        let pts: Vec<Vector3<f64>> = sol.cameras.iter().map(|c| c.mean).collect();
        assert!(planarity(&pts).unwrap().0 < 1e-9);
    }
}
