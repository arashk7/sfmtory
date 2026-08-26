//! Bag-of-visual-words image retrieval, for choosing *which* pairs to match
//! when exhaustive pairing has become too expensive.
//!
//! Exhaustive pairing is O(n^2) descriptor matches, and a descriptor match
//! between two SIFT images is itself tens of milliseconds - so it stops being
//! affordable somewhere in the low hundreds of images. Retrieval flips the
//! cost structure: describe every image once as a sparse histogram over a
//! shared visual vocabulary, score images against each other in that cheap
//! representation, and only run real descriptor matching on the handful of
//! candidates that actually look similar.
//!
//! ## Design notes
//!
//! - **The vocabulary is a hierarchical k-means tree**, as in COLMAP, not a
//!   flat codebook. A flat codebook was implemented first and measured: it
//!   works (99% recall of the pairs exhaustive matching verifies, at half the
//!   candidates) but is *slower end to end* than exhaustive matching on a
//!   47-image set, because assigning a descriptor to its nearest of `k` words
//!   costs `O(k * dim)` - 1024 * 128 multiply-adds per descriptor, paid once
//!   per training descriptor per Lloyd iteration and again for every
//!   descriptor in the dataset. Descending a tree of branching factor `b` and
//!   depth `L` costs `O(b * L * dim)` instead: for `b = 10, L = 3` that is 30
//!   centroid comparisons rather than 1000, and it is the difference between
//!   retrieval being an optimization and being a pessimization.
//! - **The vocabulary is trained on the dataset being reconstructed**, rather
//!   than loaded from a pre-trained tree shipped alongside the binary. That
//!   avoids a bundled multi-megabyte asset and the licensing question that
//!   comes with it (see PLAN.md's ground rules), and self-training is
//!   generally better for retrieval quality on the data at hand. The
//!   trade-off is that the vocabulary is not reusable across datasets and
//!   costs one training pass up front, which is why training samples a
//!   subset of descriptors rather than clustering all of them.
//!
//! Scoring uses the standard TF-IDF weighting over an inverted index: images
//! are only ever compared against images with which they share at least one
//! visual word, so the quadratic term is over *co-occurring* pairs rather
//! than all pairs.

use std::collections::HashMap;

use rayon::prelude::*;
use sfm_core::{Descriptors, FeatureSet};

/// Deterministic PRNG, same rationale as `sfm-geometry::ransac`'s: k-means
/// seeding is the only randomness here, and a fixed seed keeps `sfm match`
/// reproducible run to run - which this pipeline treats as a correctness
/// property, not a nicety (see decisions.md).
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Xorshift64(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VocabParams {
    /// Children per tree node. Leaf count (the effective vocabulary size) is
    /// at most `branching^depth`.
    pub branching: usize,
    /// Tree depth.
    pub depth: usize,
    /// Upper bound on descriptors sampled across all images for training.
    pub train_sample: usize,
    /// Lloyd iterations per node split.
    pub iterations: usize,
    /// How many retrieval candidates each image contributes.
    pub num_neighbors: usize,
    pub seed: u64,
}

impl Default for VocabParams {
    fn default() -> Self {
        VocabParams {
            // 10^3 = up to 1000 leaf words, at 30 centroid comparisons per
            // descriptor instead of 1000.
            branching: 10,
            depth: 3,
            // Enough to populate 1000 leaves without making training the
            // dominant cost - training is O(sample * branching * depth).
            train_sample: 30_000,
            iterations: 8,
            // Symmetrized below, so each image typically ends up with more
            // than this many candidates. Chosen to comfortably exceed the
            // ~10-20 genuinely co-visible neighbours a well-covered capture
            // has, since a missed pair costs a reconstruction far more than
            // an extra matched pair costs time.
            num_neighbors: 20,
            seed: 0x5EED_1234_ABCD_0001,
        }
    }
}

/// A trained hierarchical visual vocabulary. Stored as a flat arena: node `i`
/// owns centroid row `i`, and its children occupy the contiguous index range
/// `child_start[i] .. child_start[i] + child_count[i]`. Leaves have zero
/// children and carry a `word[i]` id; interior nodes have `word[i] == NO_WORD`.
struct Vocabulary {
    dim: usize,
    centroids: Vec<f32>,
    child_start: Vec<usize>,
    child_count: Vec<usize>,
    word: Vec<usize>,
    num_words: usize,
}

const NO_WORD: usize = usize::MAX;

impl Vocabulary {
    /// Descend from the root, at each level picking the nearest child
    /// centroid. This is the whole point of the tree: `branching * depth`
    /// distance evaluations instead of one per word.
    fn assign(&self, desc: &[f32]) -> usize {
        if self.centroids.is_empty() {
            return 0;
        }
        let mut node = 0usize;
        loop {
            let count = self.child_count[node];
            if count == 0 {
                return if self.word[node] == NO_WORD {
                    0
                } else {
                    self.word[node]
                };
            }
            let start = self.child_start[node];
            let mut best = start;
            let mut best_d = f32::MAX;
            for c in start..start + count {
                let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let mut d = 0.0f32;
                for t in 0..self.dim {
                    let diff = desc[t] - cen[t];
                    d += diff * diff;
                    if d >= best_d {
                        break;
                    }
                }
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            node = best;
        }
    }
}

/// One level of k-means over the given rows, returning per-row cluster
/// assignments and the `k` centroids. Seeded k-means++ then Lloyd.
fn kmeans(
    sample: &[f32],
    rows: &[usize],
    dim: usize,
    k: usize,
    iterations: usize,
    rng: &mut Xorshift64,
) -> (Vec<usize>, Vec<f32>) {
    let n = rows.len();
    let k = k.min(n).max(1);
    let row = |i: usize| -> &[f32] { &sample[rows[i] * dim..(rows[i] + 1) * dim] };

    // k-means++ seeding: spreads the initial centroids out instead of
    // clumping them, which for a vocabulary matters more than usual - a
    // clumped initialization wastes words on a dense region of descriptor
    // space and leaves the rest of it unresolved.
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    let first = rng.next_usize(n);
    centroids.extend_from_slice(row(first));
    let mut d2: Vec<f32> = (0..n)
        .map(|i| {
            row(i)
                .iter()
                .zip(&centroids[0..dim])
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        })
        .collect();
    while centroids.len() / dim < k {
        let total: f32 = d2.iter().sum();
        let pick = if total <= 0.0 {
            rng.next_usize(n)
        } else {
            let target = rng.next_f32() * total;
            let mut acc = 0.0;
            let mut chosen = n - 1;
            for (i, &v) in d2.iter().enumerate() {
                acc += v;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let base = centroids.len();
        centroids.extend_from_slice(row(pick));
        for (i, dv) in d2.iter_mut().enumerate() {
            let d: f32 = row(i)
                .iter()
                .zip(&centroids[base..base + dim])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < *dv {
                *dv = d;
            }
        }
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iterations {
        assign
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, a)| {
                let r = &sample[rows[i] * dim..(rows[i] + 1) * dim];
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for c in 0..k {
                    let cen = &centroids[c * dim..(c + 1) * dim];
                    let mut d = 0.0f32;
                    for t in 0..dim {
                        let diff = r[t] - cen[t];
                        d += diff * diff;
                        if d >= best_d {
                            break;
                        }
                    }
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                *a = best;
            });
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0usize; k];
        for (i, &a) in assign.iter().enumerate() {
            counts[a] += 1;
            let r = row(i);
            let dst = &mut sums[a * dim..(a + 1) * dim];
            for t in 0..dim {
                dst[t] += r[t];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: re-seed onto a random sample rather than
                // leaving a dead centroid nothing can ever be assigned to.
                let r = rng.next_usize(n);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(row(r));
                continue;
            }
            let inv = 1.0 / counts[c] as f32;
            for t in 0..dim {
                centroids[c * dim + t] = sums[c * dim + t] * inv;
            }
        }
    }
    (assign, centroids)
}

/// Builds the hierarchical vocabulary by recursively k-means-splitting the
/// training sample.
fn train(sample: &[f32], dim: usize, params: &VocabParams) -> Vocabulary {
    let n = sample.len().checked_div(dim).unwrap_or(0);
    let mut vocab = Vocabulary {
        dim,
        centroids: Vec::new(),
        child_start: Vec::new(),
        child_count: Vec::new(),
        word: Vec::new(),
        num_words: 0,
    };
    if n == 0 {
        return vocab;
    }
    let mut rng = Xorshift64::new(params.seed);

    // Root node: centroid unused (never compared against), children are the
    // first real split.
    vocab.centroids.extend(std::iter::repeat_n(0.0, dim));
    vocab.child_start.push(0);
    vocab.child_count.push(0);
    vocab.word.push(NO_WORD);

    // (node index, rows belonging to it, depth)
    let mut queue: Vec<(usize, Vec<usize>, usize)> = vec![(0, (0..n).collect(), 0)];
    while let Some((node, rows, depth)) = queue.pop() {
        // Stop splitting when out of depth or when a further split could not
        // give every child at least a couple of descriptors - an
        // under-populated leaf is a word that discriminates nothing.
        if depth >= params.depth || rows.len() < params.branching * 2 {
            vocab.word[node] = vocab.num_words;
            vocab.num_words += 1;
            continue;
        }
        let (assign, centroids) = kmeans(
            sample,
            &rows,
            dim,
            params.branching,
            params.iterations,
            &mut rng,
        );
        let k = centroids.len() / dim;
        let start = vocab.child_count.len();
        vocab.child_start[node] = start;
        vocab.child_count[node] = k;
        for c in 0..k {
            vocab
                .centroids
                .extend_from_slice(&centroids[c * dim..(c + 1) * dim]);
            vocab.child_start.push(0);
            vocab.child_count.push(0);
            vocab.word.push(NO_WORD);
        }
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &a) in assign.iter().enumerate() {
            buckets[a].push(rows[i]);
        }
        for (c, b) in buckets.into_iter().enumerate() {
            if b.is_empty() {
                vocab.word[start + c] = vocab.num_words;
                vocab.num_words += 1;
            } else {
                queue.push((start + c, b, depth + 1));
            }
        }
    }
    vocab
}

/// Views a feature set's descriptors as float rows, converting packed binary
/// (ORB) descriptors to 0/1 floats so one k-means path serves both. Marker
/// descriptors have no meaningful metric here and yield `None`.
fn as_float_rows(fs: &FeatureSet) -> Option<(usize, Vec<f32>)> {
    match &fs.descriptors {
        Descriptors::Float32 { dim, data } => Some((*dim as usize, data.clone())),
        Descriptors::Binary {
            bytes_per_descriptor,
            data,
        } => {
            let bytes = *bytes_per_descriptor as usize;
            let dim = bytes * 8;
            let mut out = Vec::with_capacity(fs.len() * dim);
            for i in 0..fs.len() {
                let row = &data[i * bytes..(i + 1) * bytes];
                for b in row {
                    for bit in 0..8 {
                        out.push(((b >> bit) & 1) as f32);
                    }
                }
            }
            Some((dim, out))
        }
        Descriptors::MarkerCorner { .. } => None,
    }
}

/// Retrieval-based pairing: train a vocabulary on the given features, then
/// return the union of each image's top-`num_neighbors` most similar images
/// as `(i, j)` pairs with `i < j`.
///
/// Returns `None` when retrieval doesn't apply to these descriptors (marker
/// corners), so the caller can fall back rather than silently pairing nothing.
pub fn vocab_tree_pairs(features: &[FeatureSet], params: &VocabParams) -> Option<Vec<(usize, usize)>> {
    let n = features.len();
    if n < 2 {
        return Some(Vec::new());
    }

    let rows: Vec<Option<(usize, Vec<f32>)>> = features.iter().map(as_float_rows).collect();
    if rows.iter().any(|r| r.is_none()) {
        return None;
    }
    let dim = rows.iter().flatten().map(|(d, _)| *d).max().unwrap_or(0);
    if dim == 0 || rows.iter().flatten().any(|(d, _)| *d != dim) {
        return None;
    }

    // Training sample, spread evenly across images so no single
    // feature-dense image dominates the vocabulary.
    let total_desc: usize = features.iter().map(|f| f.len()).sum();
    if total_desc == 0 {
        return Some(Vec::new());
    }
    let stride = (total_desc / params.train_sample.max(1)).max(1);
    let mut sample: Vec<f32> = Vec::new();
    for (_, data) in rows.iter().flatten() {
        let count = data.len() / dim;
        let mut i = 0;
        while i < count {
            sample.extend_from_slice(&data[i * dim..(i + 1) * dim]);
            i += stride;
        }
    }
    if sample.len() / dim < 2 {
        return Some(Vec::new());
    }

    let vocab = train(&sample, dim, params);
    let k = vocab.num_words;
    if k == 0 {
        return Some(Vec::new());
    }

    // Per-image word histograms.
    let histograms: Vec<HashMap<usize, f32>> = rows
        .par_iter()
        .map(|r| {
            let mut h: HashMap<usize, f32> = HashMap::new();
            if let Some((_, data)) = r {
                let count = data.len() / dim;
                for i in 0..count {
                    *h.entry(vocab.assign(&data[i * dim..(i + 1) * dim]))
                        .or_insert(0.0) += 1.0;
                }
            }
            h
        })
        .collect();

    // Inverted index + document frequency.
    let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (img, h) in histograms.iter().enumerate() {
        for &w in h.keys() {
            inverted[w].push(img);
        }
    }
    let idf: Vec<f32> = inverted
        .iter()
        .map(|imgs| {
            if imgs.is_empty() {
                0.0
            } else {
                (n as f32 / imgs.len() as f32).ln()
            }
        })
        .collect();

    // TF-IDF, L2-normalized so scores are cosine similarities.
    let weighted: Vec<HashMap<usize, f32>> = histograms
        .iter()
        .map(|h| {
            let total: f32 = h.values().sum::<f32>().max(1.0);
            let mut w: HashMap<usize, f32> = h
                .iter()
                .map(|(&word, &c)| (word, (c / total) * idf[word]))
                .collect();
            let norm = w.values().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in w.values_mut() {
                    *v /= norm;
                }
            }
            w
        })
        .collect();

    // Score each image only against images sharing a word with it - the
    // point of the inverted index. A word present in nearly every image
    // contributes almost nothing (idf ~ 0) yet would dominate this loop, so
    // skip those outright.
    let max_posting = (n as f32 * 0.8).ceil() as usize;
    let mut pairs: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    let scored: Vec<Vec<(usize, f32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for (&word, &wi) in &weighted[i] {
                let posting = &inverted[word];
                if posting.len() > max_posting {
                    continue;
                }
                for &j in posting {
                    if j == i {
                        continue;
                    }
                    if let Some(&wj) = weighted[j].get(&word) {
                        *acc.entry(j).or_insert(0.0) += wi * wj;
                    }
                }
            }
            let mut v: Vec<(usize, f32)> = acc.into_iter().collect();
            // Sort by score, tie-broken by index so the selection is
            // deterministic regardless of `HashMap` iteration order.
            v.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            v.truncate(params.num_neighbors);
            v
        })
        .collect();

    for (i, cands) in scored.iter().enumerate() {
        for &(j, _) in cands {
            pairs.insert((i.min(j), i.max(j)));
        }
    }
    Some(pairs.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfm_core::Keypoint;

    fn synth(seed: u64, n: usize, dim: usize, cluster: f32) -> FeatureSet {
        let mut rng = Xorshift64::new(seed);
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            for t in 0..dim {
                // Descriptors drawn around a per-image offset so images from
                // the same "cluster" look alike and different clusters don't.
                data.push(cluster + t as f32 * 0.01 + rng.next_f32() * 0.05);
            }
        }
        FeatureSet {
            keypoints: (0..n)
                .map(|_| Keypoint {
                    x: 0.0,
                    y: 0.0,
                    scale: 1.0,
                    angle: 0.0,
                    response: 1.0,
                })
                .collect(),
            descriptors: Descriptors::Float32 {
                dim: dim as u32,
                data,
            },
        }
    }

    #[test]
    fn retrieves_similar_images_over_dissimilar_ones() {
        // Two clearly separated groups; retrieval should prefer within-group
        // pairs over across-group ones.
        let mut feats = Vec::new();
        for i in 0..4 {
            feats.push(synth(1000 + i, 120, 16, 0.0));
        }
        for i in 0..4 {
            feats.push(synth(2000 + i, 120, 16, 5.0));
        }
        let params = VocabParams {
            branching: 4,
            depth: 2,
            train_sample: 5000,
            iterations: 8,
            num_neighbors: 3,
            ..Default::default()
        };
        let pairs = vocab_tree_pairs(&feats, &params).expect("float descriptors are supported");
        assert!(!pairs.is_empty());
        let cross = pairs
            .iter()
            .filter(|(i, j)| (*i < 4) != (*j < 4))
            .count();
        assert!(
            cross * 2 < pairs.len(),
            "expected mostly within-group pairs, got {cross} cross-group of {}",
            pairs.len()
        );
    }

    #[test]
    fn is_deterministic() {
        let feats: Vec<FeatureSet> = (0..5).map(|i| synth(7000 + i, 80, 16, i as f32)).collect();
        let p = VocabParams {
            branching: 3,
            depth: 2,
            train_sample: 4000,
            iterations: 5,
            num_neighbors: 2,
            ..Default::default()
        };
        let a = vocab_tree_pairs(&feats, &p).unwrap();
        let b = vocab_tree_pairs(&feats, &p).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn declines_marker_descriptors() {
        let fs = FeatureSet {
            keypoints: vec![Keypoint {
                x: 0.0,
                y: 0.0,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            }],
            descriptors: Descriptors::MarkerCorner {
                data: vec![0u8; 8],
            },
        };
        assert!(vocab_tree_pairs(&[fs.clone(), fs], &VocabParams::default()).is_none());
    }
}
