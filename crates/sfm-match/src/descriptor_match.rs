//! Descriptor matching: brute-force mutual-nearest-neighbor with Lowe's ratio
//! test for float (SIFT) and binary (ORB) descriptors, and exact-ID matching
//! for ArUco marker corners. Brute force is O(n1*n2) per image pair - not
//! replaced with an approximate/tree-based search (a kd-tree degrades to
//! near-brute-force well before 128 dimensions, which is exactly SIFT's
//! descriptor size, so it wouldn't actually help here), but the *inner*
//! per-pair distance computation matters a lot: `match_float` computes the
//! bulk squared-distance grid as one dense matrix multiply (the same
//! "quadratic expansion" trick COLMAP/FAISS use for large-scale L2 descriptor
//! matching - `|a-b|^2 = |a|^2 + |b|^2 - 2*a.b`), not a naive
//! O(n1*n2*dim) scalar loop; both float and binary matching also compute
//! each pairwise distance exactly once (see `match_float`/`match_binary`'s
//! own doc comments) rather than once for the forward pass and again for the
//! reverse cross-check.

use nalgebra::DMatrix;
use sfm_core::Descriptors;

#[derive(Debug, Clone, Copy)]
pub struct MatchParams {
    /// Lowe's ratio: a match is kept only if `best_dist < ratio * second_best_dist`.
    pub ratio_threshold: f32,
}

impl Default for MatchParams {
    fn default() -> Self {
        MatchParams {
            ratio_threshold: 0.8,
        }
    }
}

/// Putative `(idx_in_1, idx_in_2)` matches, dispatched by descriptor type.
/// Mismatched descriptor types (e.g. SIFT vs ORB) can't be compared and
/// return no matches rather than panicking.
pub fn match_descriptors(
    d1: &Descriptors,
    d2: &Descriptors,
    params: &MatchParams,
) -> Vec<(u32, u32)> {
    match (d1, d2) {
        (Descriptors::Float32 { .. }, Descriptors::Float32 { .. }) => {
            match_float(d1, d2, params.ratio_threshold)
        }
        (Descriptors::Binary { .. }, Descriptors::Binary { .. }) => {
            match_binary(d1, d2, params.ratio_threshold)
        }
        (Descriptors::MarkerCorner { .. }, Descriptors::MarkerCorner { .. }) => {
            match_marker_corners(d1, d2)
        }
        _ => Vec::new(),
    }
}

fn match_float(d1: &Descriptors, d2: &Descriptors, ratio: f32) -> Vec<(u32, u32)> {
    let (n1, n2) = (d1.len(), d2.len());
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }
    let (Descriptors::Float32 { dim, data: data1 }, Descriptors::Float32 { data: data2, .. }) =
        (d1, d2)
    else {
        return Vec::new();
    };
    let dim = *dim as usize;

    // Bulk squared distances via the quadratic expansion
    // `|a-b|^2 = |a|^2 + |b|^2 - 2*a.b`: the `a.b` term for *every* (i, j)
    // pair is one dense n1xn2 matrix multiply, not an O(n1*n2*dim) scalar
    // loop - see this module's doc comment for why that matters so much
    // more now that SIFT's default feature counts were raised to match
    // COLMAP's own (~10k/image, not ~5k; see PLAN.md's accuracy/density
    // investigation). `max(0.0)` guards against a tiny negative value from
    // floating-point cancellation when `a` and `b` are nearly identical.
    let x1 = DMatrix::from_row_slice(n1, dim, data1);
    let x2 = DMatrix::from_row_slice(n2, dim, data2);
    let dot = &x1 * x2.transpose();
    let norm1: Vec<f32> = (0..n1).map(|i| x1.row(i).norm_squared()).collect();
    let norm2: Vec<f32> = (0..n2).map(|j| x2.row(j).norm_squared()).collect();

    // Single pass over the resulting n1*n2 distance grid, updating both
    // directions' running best/second-best as we go - row `i`'s forward
    // best+second-best (for the ratio test) and column `j`'s reverse best
    // (for the mutual-NN cross-check) simultaneously, rather than a forward
    // pass followed by a full O(n1) search per match just to find its
    // reverse nearest neighbor.
    let mut best1 = vec![f32::MAX; n1];
    let mut second1 = vec![f32::MAX; n1];
    let mut best1_idx = vec![u32::MAX; n1];
    let mut best2 = vec![f32::MAX; n2];
    let mut best2_idx = vec![u32::MAX; n2];

    for i in 0..n1 {
        for j in 0..n2 {
            let dist = (norm1[i] + norm2[j] - 2.0 * dot[(i, j)]).max(0.0);
            if dist < best1[i] {
                second1[i] = best1[i];
                best1[i] = dist;
                best1_idx[i] = j as u32;
            } else if dist < second1[i] {
                second1[i] = dist;
            }
            if dist < best2[j] {
                best2[j] = dist;
                best2_idx[j] = i as u32;
            }
        }
    }

    let mut out = Vec::new();
    for i in 0..n1 {
        let j = best1_idx[i];
        if j == u32::MAX || best1[i] >= ratio * ratio * second1[i] {
            continue;
        }
        if best2_idx[j as usize] == i as u32 {
            out.push((i as u32, j));
        }
    }
    out
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn match_binary(d1: &Descriptors, d2: &Descriptors, ratio: f32) -> Vec<(u32, u32)> {
    let (n1, n2) = (d1.len(), d2.len());
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }
    // Single-pass forward+reverse nearest-neighbor computation - see
    // `match_float`'s doc comment for why this replaced two separate passes
    // over the same n1*n2 grid.
    let mut best1 = vec![u32::MAX; n1];
    let mut second1 = vec![u32::MAX; n1];
    let mut best1_idx = vec![u32::MAX; n1];
    let mut best2 = vec![u32::MAX; n2];
    let mut best2_idx = vec![u32::MAX; n2];

    for i in 0..n1 {
        let qi = d1.binary_row(i).unwrap();
        for j in 0..n2 {
            let dist = hamming(qi, d2.binary_row(j).unwrap());
            if dist < best1[i] {
                second1[i] = best1[i];
                best1[i] = dist;
                best1_idx[i] = j as u32;
            } else if dist < second1[i] {
                second1[i] = dist;
            }
            if dist < best2[j] {
                best2[j] = dist;
                best2_idx[j] = i as u32;
            }
        }
    }

    let mut out = Vec::new();
    for i in 0..n1 {
        let j = best1_idx[i];
        if j == u32::MAX || (best1[i] as f32) >= ratio * (second1[i] as f32) {
            continue;
        }
        if best2_idx[j as usize] == i as u32 {
            out.push((i as u32, j));
        }
    }
    out
}

/// Markers are matched by identity, not distance: every corner of marker
/// `m` in image 1 pairs with the same corner of marker `m` in image 2, if
/// present. Ambiguous (duplicate id+corner within one image) entries are
/// skipped rather than guessed at.
fn match_marker_corners(d1: &Descriptors, d2: &Descriptors) -> Vec<(u32, u32)> {
    use std::collections::HashMap;
    let mut index2: HashMap<(u32, u32, u32), usize> = HashMap::new();
    for j in 0..d2.len() {
        if let Some(key) = d2.marker_corner(j) {
            index2.entry(key).or_insert(j);
        }
    }
    let mut out = Vec::new();
    for i in 0..d1.len() {
        if let Some(key) = d1.marker_corner(i) {
            if let Some(&j) = index2.get(&key) {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_set(rows: &[[f32; 4]]) -> Descriptors {
        Descriptors::Float32 {
            dim: 4,
            data: rows.iter().flatten().copied().collect(),
        }
    }

    #[test]
    fn matches_identical_float_descriptors() {
        let d1 = float_set(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
        let d2 = float_set(&[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]);
        let matches = match_descriptors(&d1, &d2, &MatchParams::default());
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 1)));
        assert!(matches.contains(&(1, 0)));
    }

    #[test]
    fn rejects_ambiguous_matches_via_ratio_test() {
        // Two equally-close target descriptors make the best match ambiguous
        // (neither is a confident nearest neighbor over the other).
        let d1 = float_set(&[[1.0, 0.0, 0.0, 0.0]]);
        let d2 = float_set(&[[0.9, 0.1, 0.0, 0.0], [0.89, 0.11, 0.0, 0.0]]);
        let matches = match_descriptors(
            &d1,
            &d2,
            &MatchParams {
                ratio_threshold: 0.8,
            },
        );
        assert!(
            matches.is_empty(),
            "ambiguous match should be rejected: {matches:?}"
        );
    }

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
    fn matches_marker_corners_by_identity() {
        let d1 = corners(&[(0, 3, 0), (0, 3, 1)]);
        let d2 = corners(&[(0, 3, 1), (0, 3, 0)]);
        let matches = match_descriptors(&d1, &d2, &MatchParams::default());
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 1)));
        assert!(matches.contains(&(1, 0)));
    }

    #[test]
    fn marker_corners_do_not_match_across_captures() {
        // The same physical marker, photographed in two captures between
        // which it was moved: matching these would invent a correspondence
        // between two unrelated 3D locations.
        let d1 = corners(&[(0, 3, 0), (0, 3, 1)]);
        let d2 = corners(&[(1, 3, 0), (1, 3, 1)]);
        assert!(match_descriptors(&d1, &d2, &MatchParams::default()).is_empty());
    }

    #[test]
    fn marker_corners_match_within_a_capture_across_cameras() {
        // Two cameras seeing the same marker in the same capture must match -
        // this is what makes a multi-camera rig reconstructable at all.
        let d1 = corners(&[(2, 7, 0), (2, 7, 2)]);
        let d2 = corners(&[(2, 7, 2), (2, 7, 0)]);
        let matches = match_descriptors(&d1, &d2, &MatchParams::default());
        assert_eq!(matches.len(), 2);
    }
}
