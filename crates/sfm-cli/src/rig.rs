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
    /// `(capture, cameras that agreed, cameras shared)` per aligned capture.
    /// A low ratio means that capture's reconstruction disagrees with the
    /// reference about where most of the rig is, which is worth seeing.
    pub agreement: Vec<(i64, usize, usize)>,
}

/// Similarity alignment that tolerates a minority of badly-placed cameras.
///
/// Plain least squares is the wrong tool here and measurably so. On a
/// 193-camera cube rig every capture reconstructed 189-192 cameras, but a
/// handful in each came out grossly misplaced; fitting all of them at once
/// dragged the whole transform, and the aligned result had a mean spread of
/// **124% of camera spacing** with individual cameras past 1000%. A single bad
/// correspondence in a least-squares similarity moves the answer for every
/// other camera.
///
/// So the transform is estimated from minimal samples and scored by how many
/// cameras it explains, then refitted on those. `threshold` is in the units of
/// `to`, and callers should scale it to the rig rather than pass an absolute.
pub fn robust_umeyama(
    from: &[Vector3<f64>],
    to: &[Vector3<f64>],
    threshold: f64,
) -> Option<(Similarity, Vec<usize>)> {
    let n = from.len();
    if n < 3 {
        return None;
    }
    // Deterministic sampling: a calibration that changes between identical
    // runs cannot be checked against itself.
    let mut best: Option<(Similarity, Vec<usize>)> = None;
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    const ITERATIONS: usize = 256;
    for _ in 0..ITERATIONS {
        let mut idx = [0usize; 3];
        for slot in idx.iter_mut() {
            *slot = (next() % n as u64) as usize;
        }
        if idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2] {
            continue;
        }
        let f: Vec<Vector3<f64>> = idx.iter().map(|&i| from[i]).collect();
        let t: Vec<Vector3<f64>> = idx.iter().map(|&i| to[i]).collect();
        let Some(sim) = umeyama(&f, &t) else {
            continue;
        };
        if !sim.scale.is_finite() || sim.scale <= 0.0 {
            continue;
        }
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| (sim.apply(&from[i]) - to[i]).norm() < threshold)
            .collect();
        if best.as_ref().is_none_or(|(_, b)| inliers.len() > b.len()) {
            best = Some((sim, inliers));
        }
    }
    let (_, inliers) = best?;
    if inliers.len() < 3 {
        return None;
    }
    // Refit on everything the winning sample explained.
    let f: Vec<Vector3<f64>> = inliers.iter().map(|&i| from[i]).collect();
    let t: Vec<Vector3<f64>> = inliers.iter().map(|&i| to[i]).collect();
    umeyama(&f, &t).map(|sim| (sim, inliers))
}

/// Brings every capture into the reference capture's frame and reports where
/// each camera sits, with its spread.
pub fn solve(rigs: &[CaptureRig], min_shared: usize) -> Result<RigSolution> {
    // Reference by quality, not just by count. Picking purely the most
    // cameras chose a capture with 5.6px reprojection error over four others
    // between 0.66 and 1.35 - every other capture was then aligned onto the
    // worst reconstruction available.
    let mut errs: Vec<f64> = rigs
        .iter()
        .filter(|r| r.mean_reprojection_px.is_finite() && r.mean_reprojection_px > 0.0)
        .map(|r| r.mean_reprojection_px)
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_err = errs.get(errs.len() / 2).copied().unwrap_or(f64::MAX);
    let acceptable = median_err * 2.0;
    let reference = rigs
        .iter()
        .filter(|r| r.mean_reprojection_px <= acceptable)
        .max_by_key(|r| r.centres.len())
        .or_else(|| rigs.iter().max_by_key(|r| r.centres.len()));
    let Some(reference) = reference else {
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
    let mut agreement: Vec<(i64, usize, usize)> = Vec::new();
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
        // Threshold relative to the reference rig's own size, so it means the
        // same thing whatever arbitrary scale this reconstruction came out at.
        let mut spread = 0.0;
        let mut count = 0usize;
        for i in 0..to.len() {
            for j in (i + 1)..to.len() {
                spread += (to[i] - to[j]).norm();
                count += 1;
            }
        }
        let mean_spacing = if count > 0 {
            spread / count as f64
        } else {
            1.0
        };
        let Some((sim, inliers)) = robust_umeyama(&from, &to, 0.15 * mean_spacing) else {
            skipped.push((rig.capture_id, "cameras are degenerate (collinear?)".into()));
            continue;
        };
        agreement.push((rig.capture_id, inliers.len(), shared.len()));
        // Only cameras the transform actually explains contribute. A camera
        // this capture placed badly should not move the average for it.
        let inlier_ids: std::collections::BTreeSet<u32> =
            inliers.iter().map(|&i| shared[i]).collect();
        for (id, c) in &rig.centres {
            if !inlier_ids.contains(id) {
                continue;
            }
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
        agreement,
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

    /// The failure that made scan2's alignment useless: a handful of grossly
    /// misplaced cameras dragging a least-squares fit for every other camera.
    #[test]
    fn robust_alignment_ignores_grossly_misplaced_cameras() {
        let truth: Vec<Vector3<f64>> = (0..20)
            .map(|i| {
                let a = i as f64 * 0.31;
                v(a.cos(), a.sin(), 0.1 * (i % 4) as f64)
            })
            .collect();
        let angle = 0.4_f64;
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
        let (scale, t) = (1.7, v(2.0, -3.0, 1.0));
        let mut moved: Vec<Vector3<f64>> = truth.iter().map(|p| scale * (rot * p) + t).collect();
        // Four of twenty come out somewhere else entirely.
        for k in [2usize, 7, 11, 18] {
            moved[k] = v(40.0 + k as f64, -30.0, 25.0);
        }

        let plain = umeyama(&truth, &moved).unwrap();
        let plain_err: f64 = truth
            .iter()
            .zip(&moved)
            .enumerate()
            .filter(|(i, _)| ![2, 7, 11, 18].contains(i))
            .map(|(_, (f, t))| (plain.apply(f) - t).norm())
            .sum();

        let (robust, inliers) = robust_umeyama(&truth, &moved, 0.15 * 2.0).unwrap();
        let robust_err: f64 = truth
            .iter()
            .zip(&moved)
            .enumerate()
            .filter(|(i, _)| ![2, 7, 11, 18].contains(i))
            .map(|(_, (f, t))| (robust.apply(f) - t).norm())
            .sum();

        assert_eq!(inliers.len(), 16, "should keep exactly the 16 good cameras");
        for bad in [2usize, 7, 11, 18] {
            assert!(!inliers.contains(&bad), "kept a misplaced camera {bad}");
        }
        assert!(
            robust_err < plain_err * 1e-3,
            "robust {robust_err} should be far below least-squares {plain_err}"
        );
        assert!((robust.scale - scale).abs() < 1e-6);
    }

    #[test]
    fn reference_capture_is_chosen_by_quality_not_only_by_count() {
        let centres = |n: u32| -> BTreeMap<u32, Vector3<f64>> {
            (0..n)
                .map(|i| (i, v(i as f64, 0.0, (i % 3) as f64)))
                .collect()
        };
        let rigs = vec![
            // Most cameras, but a badly-converged reconstruction.
            CaptureRig {
                capture_id: 16,
                centres: centres(6),
                rotations: BTreeMap::new(),
                num_points: 10,
                mean_reprojection_px: 5.6,
            },
            CaptureRig {
                capture_id: 9,
                centres: centres(5),
                rotations: BTreeMap::new(),
                num_points: 10,
                mean_reprojection_px: 0.7,
            },
            CaptureRig {
                capture_id: 23,
                centres: centres(5),
                rotations: BTreeMap::new(),
                num_points: 10,
                mean_reprojection_px: 0.8,
            },
        ];
        let sol = solve(&rigs, 3).unwrap();
        assert_ne!(
            sol.reference, 16,
            "the 5.6px capture must not be the reference"
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
