pub mod descriptor_match;
pub mod pairing;
pub mod vocab;

use sfm_core::{CameraModel, FeatureSet, TwoViewGeometryRecord};
use sfm_geometry::{estimate_two_view_geometry, to_normalized};

pub use descriptor_match::{match_descriptors, MatchParams};
pub use pairing::{exhaustive_pairs, sequential_pairs};
pub use vocab::{vocab_tree_pairs, VocabParams};

#[derive(Debug, Clone, Copy)]
pub struct VerificationParams {
    pub match_params: MatchParams,
    /// RANSAC inlier threshold, in pixels.
    pub ransac_threshold_px: f64,
    pub ransac_max_iterations: usize,
    /// A pair is rejected outright (kept out of the reconstruction) unless
    /// it clears both of these - the main defense against a coincidentally-
    /// matched but geometrically-nonsensical pair polluting the reconstruction.
    ///
    /// `min_inlier_ratio` in particular needs to stay low: architecture
    /// (repeated windows, columns, symmetric facade details - exactly what
    /// `data/sceaux_castle` is) generates large numbers of confident-looking
    /// but geometrically-wrong SIFT matches between similar-but-different
    /// repeated elements, which drags the *ratio* down even when the
    /// absolute inlier count is high and the pose is solid. A real pair with
    /// 83-133 correct RANSAC inliers got rejected here at the previous
    /// default of 0.25 purely on ratio, costing real registrations - see
    /// PLAN.md's real-data-testing entry. Now matches COLMAP's own default
    /// exactly (`TwoViewGeometryOptions::min_inlier_ratio = 0.0`, confirmed
    /// via `pycolmap`): no ratio gate at all, `min_inliers` alone (also
    /// COLMAP's own default value, 15) is the entire defense. An even lower
    /// non-zero floor was tried as a "sanity backstop" against a degenerate
    /// handful-of-coincidental-matches case, but measurably cost verified
    /// pairs relative to COLMAP on real data without evidence it ever
    /// caught a real bad pair - removed rather than kept on spec.
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
}

impl Default for VerificationParams {
    fn default() -> Self {
        VerificationParams {
            match_params: MatchParams::default(),
            ransac_threshold_px: 4.0,
            ransac_max_iterations: 2000,
            min_inliers: 15,
            min_inlier_ratio: 0.0,
        }
    }
}

/// Match two images' features and geometrically verify the result. Returns
/// `None` if too few putative matches exist, or too few survive RANSAC to
/// trust the pair (see `VerificationParams::min_inliers`/`min_inlier_ratio`) -
/// callers should simply not store the pair in that case, rather than storing
/// a low-confidence geometry.
pub fn match_and_verify(
    features1: &FeatureSet,
    features2: &FeatureSet,
    cam1: &CameraModel,
    cam2: &CameraModel,
    params: &VerificationParams,
) -> Option<TwoViewGeometryRecord> {
    let putative = match_descriptors(
        &features1.descriptors,
        &features2.descriptors,
        &params.match_params,
    );
    if putative.len() < params.min_inliers.max(8) {
        return None;
    }

    let pts1: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(i, _)| {
            let kp = features1.keypoints[i as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();
    let pts2: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(_, j)| {
            let kp = features2.keypoints[j as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();

    // Fiducial corners take the same geometric-verification path as everything
    // else. An earlier version short-circuited here and returned
    // `Pose::identity()` with all putative matches accepted, on the reasoning
    // that exact-ID marker correspondences are already outlier-free so RANSAC
    // has nothing to reject. That reasoning is right about *outliers* and
    // misses what the step is also for: `estimate_two_view_geometry` is where
    // the relative **pose** comes from. Returning an identity pose stored a
    // zero baseline for every fiducial pair, so the reconstruction bootstrapped
    // with both cameras at the same point and triangulated every corner to the
    // origin - fiducial-only datasets could not reconstruct at all. The
    // `min_inliers` gate (15 by default, i.e. at least four markers) already
    // keeps pairs too thin for a stable eight-point estimate from reaching
    // here.

    // Fiducial corners carry the capture they were seen in, and after
    // `--merge-multicaps` one image holds several. Verify each capture on its
    // own and pool the results: see `verify_per_group` for why one pass over
    // the union throws away a third of the correct matches.
    let groups = capture_groups(&features1.descriptors, &putative);
    let geometry = match groups {
        Some(g) => verify_per_group(&pts1, &pts2, cam1, cam2, params, &g)?,
        None => estimate_two_view_geometry(
            &pts1,
            &pts2,
            cam1,
            cam2,
            params.ransac_threshold_px,
            params.ransac_max_iterations,
        )?,
    };

    let inlier_ratio = geometry.num_inliers as f64 / putative.len() as f64;
    if geometry.num_inliers < params.min_inliers || inlier_ratio < params.min_inlier_ratio {
        return None;
    }

    let inlier_matches: Vec<(u32, u32)> = putative
        .iter()
        .zip(geometry.inliers.iter())
        .filter_map(|(&m, &keep)| keep.then_some(m))
        .collect();

    Some(TwoViewGeometryRecord {
        pose: geometry.pose,
        inlier_matches,
    })
}

/// The capture each putative match belongs to, when the descriptors carry one.
///
/// `None` for anything but fiducial corners, and for a set that turns out to
/// come from a single capture - both of which the ordinary single-pass path
/// already handles correctly.
fn capture_groups(d1: &sfm_core::Descriptors, putative: &[(u32, u32)]) -> Option<Vec<u32>> {
    let sfm_core::Descriptors::MarkerCorner { .. } = d1 else {
        return None;
    };
    let groups: Vec<u32> = putative
        .iter()
        .map(|&(i, _)| d1.marker_corner(i as usize).map(|(c, _, _)| c))
        .collect::<Option<Vec<u32>>>()?;
    let distinct: std::collections::BTreeSet<u32> = groups.iter().copied().collect();
    (distinct.len() > 1).then_some(groups)
}

/// Verifies each rigid group separately, then refits one pose on the union.
///
/// A merged pair's correspondences are not one scene. Each capture is a rigid
/// group - the target was moved between them - and for a marker board each
/// group is *planar*, which `estimate_two_view_geometry` handles explicitly by
/// preferring a homography where the essential matrix is unidentifiable. The
/// union of several such groups is neither case: no homography covers more
/// than one capture's worth, and the eight-point essential RANSAC is drawing
/// minimal samples across a union of planes. There is no branch for that, so
/// it settles for whichever model explains about half.
///
/// Measured on a 4-camera rig, same camera pair, only the merge differing:
/// one capture kept 215 of 308 putative matches (70%); the same cameras merged
/// kept 386 of 800 (48%), losing 29% of the project's correspondences and with
/// them the multi-view tracks that the rig geometry depends on.
///
/// The relative pose is identical for every group - the cameras did not move -
/// so the per-group inlier sets are directly combinable, and the pose is
/// refitted on the union afterwards. That final fit is better conditioned than
/// any group's: several planes at different orientations are exactly the
/// configuration a single plane fails to provide.
fn verify_per_group(
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    cam1: &CameraModel,
    cam2: &CameraModel,
    params: &VerificationParams,
    groups: &[u32],
) -> Option<sfm_geometry::TwoViewGeometry> {
    use std::collections::BTreeMap;
    let mut by_group: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        by_group.entry(*g).or_default().push(i);
    }

    let mut inliers = vec![false; pts1.len()];
    let mut any = false;
    for idx in by_group.values() {
        if idx.len() < 8 {
            continue;
        }
        let g1: Vec<(f64, f64)> = idx.iter().map(|&i| pts1[i]).collect();
        let g2: Vec<(f64, f64)> = idx.iter().map(|&i| pts2[i]).collect();
        let Some(geom) = estimate_two_view_geometry(
            &g1,
            &g2,
            cam1,
            cam2,
            params.ransac_threshold_px,
            params.ransac_max_iterations,
        ) else {
            continue;
        };
        for (k, &keep) in geom.inliers.iter().enumerate() {
            if keep {
                inliers[idx[k]] = true;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }

    // Refit the pose on everything the groups agreed on. The union spans
    // several planes, so the essential matrix is identifiable here even though
    // it was not within any single group.
    let keep_idx: Vec<usize> = (0..pts1.len()).filter(|&i| inliers[i]).collect();
    if keep_idx.len() < 8 {
        return None;
    }
    let u1: Vec<(f64, f64)> = keep_idx.iter().map(|&i| pts1[i]).collect();
    let u2: Vec<(f64, f64)> = keep_idx.iter().map(|&i| pts2[i]).collect();
    let pooled = estimate_two_view_geometry(
        &u1,
        &u2,
        cam1,
        cam2,
        params.ransac_threshold_px,
        params.ransac_max_iterations,
    )?;

    // The union is the inlier set; the pooled fit supplies only the pose.
    //
    // Re-filtering against the pooled model was tried and lost matches rather
    // than gaining them - 1588 against 1933 on the merged rig - because it
    // applies the ill-conditioned union fit a second time and intersects two
    // rejections. Each group already verified its own correspondences against
    // a model that suits it, which is the entire point of splitting them.
    Some(sfm_geometry::TwoViewGeometry {
        pose: pooled.pose,
        inliers,
        num_inliers: keep_idx.len(),
    })
}

/// Convenience used by callers that already have pixel points in hand
/// (kept in sync with [`sfm_geometry::essential::to_normalized`] so
/// `sfm-reconstruction` doesn't need its own copy).
pub fn normalize_point(p: (f64, f64), cam: &CameraModel) -> (f64, f64) {
    to_normalized(p, cam)
}

#[cfg(test)]
mod group_tests {
    use super::*;
    use sfm_core::Descriptors;

    fn corners(rows: &[(u32, u32, u32)]) -> Descriptors {
        let mut data = Vec::new();
        for &(capture, marker, corner) in rows {
            data.extend_from_slice(&capture.to_le_bytes());
            data.extend_from_slice(&marker.to_le_bytes());
            data.extend_from_slice(&corner.to_le_bytes());
        }
        Descriptors::MarkerCorner { data }
    }

    #[test]
    fn several_captures_are_grouped_by_capture() {
        let d = corners(&[(9, 1, 0), (9, 1, 1), (16, 1, 0), (16, 1, 1)]);
        let putative = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        assert_eq!(
            capture_groups(&d, &putative),
            Some(vec![9, 9, 16, 16]),
            "a merged pair must be split along its captures"
        );
    }

    /// The ordinary paths must be left exactly as they were: one capture is
    /// already handled correctly, and splitting it would only remove points
    /// from a fit that needs them.
    #[test]
    fn a_single_capture_and_non_marker_descriptors_are_not_grouped() {
        let one = corners(&[(9, 1, 0), (9, 1, 1), (9, 2, 0)]);
        assert_eq!(capture_groups(&one, &[(0, 0), (1, 1), (2, 2)]), None);

        let sift = Descriptors::Float32 {
            dim: 2,
            data: vec![0.0; 6],
        };
        assert_eq!(capture_groups(&sift, &[(0, 0), (1, 1), (2, 2)]), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use sfm_core::{Keypoint, Pose};

    fn pinhole(f: f64, w: u32, h: u32) -> CameraModel {
        CameraModel::SimplePinhole {
            f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }

    /// A distinct, unambiguous descriptor per index (real SIFT/ORB descriptors
    /// are never exact duplicates across genuinely different keypoints, so
    /// matching them doesn't need to cope with that degenerate case here).
    fn descriptor_row(seed: usize, dim: usize) -> Vec<f32> {
        let mut row = vec![0f32; dim];
        row[seed] = 1.0;
        row
    }

    #[test]
    fn end_to_end_match_and_verify_on_synthetic_two_view() {
        let cam = pinhole(800.0, 640, 480);
        let true_translation = Vector3::new(1.0, 0.0, 0.0);
        let true_rotation = UnitQuaternion::from_euler_angles(0.0, 0.15, 0.0);
        let pose2_true = Pose::from_rotation_translation(true_rotation, true_translation);

        let points_3d: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new(0.6 * (t * 0.9).sin(), 0.4 * (t * 1.3).cos(), 4.0 + 0.07 * t)
            })
            .collect();

        let mut kps1 = Vec::new();
        let mut kps2 = Vec::new();
        let mut desc1 = Vec::new();
        let mut desc2 = Vec::new();
        for (i, p) in points_3d.iter().enumerate() {
            let (x1, y1) = cam.project(p);
            let (x2, y2) = cam.project(&pose2_true.transform_point(p));
            kps1.push(Keypoint {
                x: x1 as f32,
                y: y1 as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            });
            kps2.push(Keypoint {
                x: x2 as f32,
                y: y2 as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            });
            desc1.extend_from_slice(&descriptor_row(i, points_3d.len()));
            desc2.extend_from_slice(&descriptor_row(i, points_3d.len()));
        }

        let dim = points_3d.len() as u32;
        let features1 = FeatureSet {
            keypoints: kps1,
            descriptors: sfm_core::Descriptors::Float32 { dim, data: desc1 },
        };
        let features2 = FeatureSet {
            keypoints: kps2,
            descriptors: sfm_core::Descriptors::Float32 { dim, data: desc2 },
        };

        let record = match_and_verify(
            &features1,
            &features2,
            &cam,
            &cam,
            &VerificationParams::default(),
        )
        .expect("should verify a clean synthetic two-view pair");
        assert!(
            record.inlier_matches.len() >= 35,
            "got {} inliers",
            record.inlier_matches.len()
        );

        let cos_angle = record
            .pose
            .translation
            .normalize()
            .dot(&true_translation.normalize())
            .abs();
        assert!(cos_angle > 0.99, "cos_angle={cos_angle}");
    }
}
