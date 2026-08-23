//! Generic adaptive RANSAC: draws minimal samples, fits a model, scores every
//! point's residual, and shrinks the required iteration count as the best
//! inlier ratio improves. Used for both two-view essential-matrix estimation
//! and PnP camera registration - the model/residual are supplied by the
//! caller so this file has no domain knowledge of either.
//!
//! Uses a small seeded xorshift PRNG instead of the `rand` crate: RANSAC
//! sampling doesn't need cryptographic or even particularly high-quality
//! randomness, and a fixed seed makes `sfm map` runs reproducible (same
//! input images always give the same reconstruction), which matters for a
//! calibration tool people will want to trust and diff between runs.

pub struct RansacParams {
    pub max_iterations: usize,
    /// Residual threshold below which a point counts as an inlier, in
    /// whatever units the caller's residual function returns.
    pub threshold: f64,
    pub confidence: f64,
}

impl Default for RansacParams {
    fn default() -> Self {
        RansacParams {
            max_iterations: 2000,
            threshold: 1.0,
            confidence: 0.999,
        }
    }
}

impl RansacParams {
    pub fn with_threshold(threshold: f64) -> Self {
        RansacParams {
            threshold,
            ..Default::default()
        }
    }
}

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
    fn next_below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Fisher-Yates partial shuffle: pick `k` distinct indices from `0..n`.
fn sample_indices(rng: &mut Xorshift64, n: usize, k: usize, buf: &mut Vec<usize>) {
    buf.clear();
    buf.extend(0..n);
    let take = k.min(n);
    for i in 0..take {
        let j = i + rng.next_below(n - i);
        buf.swap(i, j);
    }
    buf.truncate(take);
}

/// Runs RANSAC and returns `(best_model, inlier_mask)`, or `None` if no
/// sample ever produced a valid model. `estimate` receives the sampled point
/// indices and returns a candidate model; `residual` scores how well a given
/// point fits a given model (same units as `params.threshold`).
pub fn ransac<M, EstimateFn, ResidualFn>(
    n: usize,
    sample_size: usize,
    params: &RansacParams,
    seed: u64,
    estimate: EstimateFn,
    residual: ResidualFn,
) -> Option<(M, Vec<bool>)>
where
    EstimateFn: Fn(&[usize]) -> Option<M>,
    ResidualFn: Fn(&M, usize) -> f64,
{
    if n < sample_size {
        return None;
    }
    let mut rng = Xorshift64::new(seed);
    let mut sample_buf = Vec::with_capacity(sample_size);
    let mut best: Option<(M, Vec<bool>, usize)> = None;
    let mut needed_iterations = params.max_iterations;

    let mut iter = 0;
    while iter < params.max_iterations.min(needed_iterations.max(1)) {
        iter += 1;
        sample_indices(&mut rng, n, sample_size, &mut sample_buf);
        let model = match estimate(&sample_buf) {
            Some(m) => m,
            None => continue,
        };
        let mut inliers = Vec::with_capacity(n);
        let mut count = 0;
        for i in 0..n {
            let ok = residual(&model, i) < params.threshold;
            inliers.push(ok);
            if ok {
                count += 1;
            }
        }
        if best.as_ref().map(|(_, _, c)| count > *c).unwrap_or(true) {
            best = Some((model, inliers, count));
            let inlier_ratio = (count as f64 / n as f64).max(1e-6);
            let denom = (1.0 - inlier_ratio.powi(sample_size as i32))
                .max(1e-12)
                .ln();
            if denom < 0.0 {
                let estimate = ((1.0 - params.confidence).ln() / denom).ceil();
                if estimate.is_finite() && estimate >= 0.0 {
                    needed_iterations = (estimate as usize).min(params.max_iterations);
                }
            }
        }
    }

    best.map(|(m, inliers, _)| (m, inliers))
}
