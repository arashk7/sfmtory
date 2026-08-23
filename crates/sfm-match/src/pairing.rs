//! Image-pair generation strategies. `vocab-tree`, `spatial`, and `aruco`
//! (shared-marker co-visibility) pairing are still TODO (see PLAN.md) - both
//! are essential for scaling past a few hundred images, since exhaustive
//! pairing is O(n^2).

/// All `(i, j)` with `i < j` - correct for any image count but O(n^2) pairs,
/// so only appropriate for small (roughly <300 image) unordered collections.
pub fn exhaustive_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(n.saturating_sub(1) * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// `(i, j)` for `j` within `window` positions after `i` in the given (e.g.
/// filename-sorted, so usually capture-time-ordered) sequence - the right
/// choice for video frames or a walked capture path, O(n * window).
pub fn sequential_pairs(n: usize, window: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..=(i + window).min(n.saturating_sub(1)) {
            if j < n {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_covers_all_unordered_pairs() {
        let pairs = exhaustive_pairs(4);
        assert_eq!(pairs.len(), 6);
        assert!(pairs.contains(&(0, 3)));
    }

    #[test]
    fn sequential_respects_window_and_bounds() {
        let pairs = sequential_pairs(5, 2);
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(0, 2)));
        assert!(!pairs.contains(&(0, 3)));
        assert!(pairs.iter().all(|&(i, j)| j < 5 && i < j));
    }
}
