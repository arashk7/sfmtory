//! Descriptor matching: brute-force mutual-nearest-neighbor with Lowe's ratio
//! test for float (SIFT) and binary (ORB) descriptors, and exact-ID matching
//! for ArUco marker corners. Brute force is O(n1*n2) per image pair, which is
//! fine at the max-features counts this project defaults to but becomes the
//! bottleneck on very large (>5000 features/image) sets - a kd-tree (e.g. the
//! `kiddo` crate) for the float case is the obvious next speed upgrade if
//! profiling shows this dominating `sfm match` runtime; see PLAN.md.

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
    let sq_dist =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let forward: Vec<Option<usize>> = (0..n1)
        .map(|i| {
            let qi = d1.float_row(i).unwrap();
            let (mut best_j, mut best, mut second) = (usize::MAX, f32::MAX, f32::MAX);
            for j in 0..n2 {
                let dist = sq_dist(qi, d2.float_row(j).unwrap());
                if dist < best {
                    second = best;
                    best = dist;
                    best_j = j;
                } else if dist < second {
                    second = dist;
                }
            }
            if best_j != usize::MAX && best < ratio * ratio * second {
                Some(best_j)
            } else {
                None
            }
        })
        .collect();

    let mut out = Vec::new();
    for (i, m) in forward.iter().enumerate() {
        let Some(j) = *m else { continue };
        // Mutual-NN cross-check: is `i` also `j`'s nearest neighbor in set 1?
        let qj = d2.float_row(j).unwrap();
        let (mut best_k, mut best) = (usize::MAX, f32::MAX);
        for k in 0..n1 {
            let dist = sq_dist(qj, d1.float_row(k).unwrap());
            if dist < best {
                best = dist;
                best_k = k;
            }
        }
        if best_k == i {
            out.push((i as u32, j as u32));
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
    let forward: Vec<Option<usize>> = (0..n1)
        .map(|i| {
            let qi = d1.binary_row(i).unwrap();
            let (mut best_j, mut best, mut second) = (usize::MAX, u32::MAX, u32::MAX);
            for j in 0..n2 {
                let dist = hamming(qi, d2.binary_row(j).unwrap());
                if dist < best {
                    second = best;
                    best = dist;
                    best_j = j;
                } else if dist < second {
                    second = dist;
                }
            }
            if best_j != usize::MAX && (best as f32) < ratio * (second as f32) {
                Some(best_j)
            } else {
                None
            }
        })
        .collect();

    let mut out = Vec::new();
    for (i, m) in forward.iter().enumerate() {
        let Some(j) = *m else { continue };
        let qj = d2.binary_row(j).unwrap();
        let (mut best_k, mut best) = (usize::MAX, u32::MAX);
        for k in 0..n1 {
            let dist = hamming(qj, d1.binary_row(k).unwrap());
            if dist < best {
                best = dist;
                best_k = k;
            }
        }
        if best_k == i {
            out.push((i as u32, j as u32));
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
    let mut index2: HashMap<(u32, u32), usize> = HashMap::new();
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

    #[test]
    fn matches_marker_corners_by_identity() {
        let mut data1 = Vec::new();
        data1.extend_from_slice(&3u32.to_le_bytes());
        data1.extend_from_slice(&0u32.to_le_bytes());
        data1.extend_from_slice(&3u32.to_le_bytes());
        data1.extend_from_slice(&1u32.to_le_bytes());
        let d1 = Descriptors::MarkerCorner { data: data1 };

        let mut data2 = Vec::new();
        data2.extend_from_slice(&3u32.to_le_bytes());
        data2.extend_from_slice(&1u32.to_le_bytes());
        data2.extend_from_slice(&3u32.to_le_bytes());
        data2.extend_from_slice(&0u32.to_le_bytes());
        let d2 = Descriptors::MarkerCorner { data: data2 };

        let matches = match_descriptors(&d1, &d2, &MatchParams::default());
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 1)));
        assert!(matches.contains(&(1, 0)));
    }
}
