//! Focal-length estimation from orthogonal vanishing points.
//!
//! In a scene built from mutually perpendicular structure - walls, tables,
//! screens, buildings, almost any man-made environment - three families of
//! parallel 3D lines project to three vanishing points. For a camera with
//! square pixels and zero skew, any *two* vanishing points from perpendicular
//! directions constrain the focal length directly:
//!
//! ```text
//! (v1 - p) . (v2 - p) + f^2 = 0
//! ```
//!
//! where `p` is the principal point. The derivation is short: a vanishing
//! point is the image of a direction, so `d ~ K^-1 v`, and perpendicular
//! directions have `d1 . d2 = 0`, which expands to the above once `K` is
//! `diag(f, f, 1)` about `p`.
//!
//! This needs no training data, no download, and no scene metadata - only that
//! the scene contains perpendicular lines. When it doesn't, the estimator
//! abstains rather than guessing: the whole point of the surrounding cascade
//! is that a technique which cannot answer says so.

use image::GrayImage;
use imageproc::hough::{detect_lines, LineDetectionOptions, PolarLine};

/// A line in homogeneous form `ax + by + c = 0`, from a Hough polar line.
fn line_coeffs(l: &PolarLine) -> (f64, f64, f64) {
    let t = (l.angle_in_degrees as f64).to_radians();
    (t.cos(), t.sin(), -(l.r as f64))
}

/// Intersection of two homogeneous lines, or `None` when near-parallel.
fn intersect(a: (f64, f64, f64), b: (f64, f64, f64)) -> Option<(f64, f64)> {
    let (x, y, w) = (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    );
    // A tiny `w` means the lines are (near) parallel, so the vanishing point
    // is at infinity - real, but useless as a numeric constraint here.
    if w.abs() < 1e-9 {
        return None;
    }
    let (px, py) = (x / w, y / w);
    if !px.is_finite() || !py.is_finite() {
        return None;
    }
    Some((px, py))
}

/// Perpendicular distance from a point to a line whose coefficients are
/// already normalized (`a^2 + b^2 = 1`, which polar form guarantees).
fn point_line_distance(l: (f64, f64, f64), p: (f64, f64)) -> f64 {
    (l.0 * p.0 + l.1 * p.1 + l.2).abs()
}

struct Xorshift64(u64);
impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// A line segment recovered from the edge map: the portion of a Hough line
/// that edge pixels actually support.
#[derive(Clone, Copy)]
struct Segment {
    a: (f64, f64),
    b: (f64, f64),
    line: (f64, f64, f64),
}

impl Segment {
    /// Whether `vp` is plausibly this segment's vanishing point: the segment's
    /// line must pass near it, *and* the point must lie beyond one of the
    /// endpoints rather than within the segment's own extent.
    ///
    /// That second condition is the one that matters. Without it, any two
    /// lines crossing near the middle of the image look like a vanishing
    /// point, and since most detected lines pass through the middle, RANSAC
    /// reliably converges on that accidental concurrence instead of on real
    /// scene structure - observed on both real photos and synthetic scenes
    /// with known focal length, where every candidate clustered within a
    /// couple of hundred pixels of the principal point and yielded no
    /// physically possible focal at all.
    fn supports(&self, vp: (f64, f64), tol: f64) -> bool {
        if point_line_distance(self.line, vp) > tol {
            return false;
        }
        let dx = self.b.0 - self.a.0;
        let dy = self.b.1 - self.a.1;
        let len2 = dx * dx + dy * dy;
        if len2 < 1e-9 {
            return false;
        }
        // Position of `vp` along the segment, 0 at `a` and 1 at `b`.
        let t = ((vp.0 - self.a.0) * dx + (vp.1 - self.a.1) * dy) / len2;
        !(-0.2..=1.2).contains(&t)
    }
}

/// Recovers the supported extent of each Hough line by walking it through the
/// edge map and keeping the longest run of nearby edge pixels.
fn segments_from_edges(edges: &GrayImage, lines: &[PolarLine], min_len: f64) -> Vec<Segment> {
    let (w, h) = edges.dimensions();
    let is_edge = |x: f64, y: f64| -> bool {
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (px, py) = (xi + dx, yi + dy);
                if px >= 0
                    && py >= 0
                    && (px as u32) < w
                    && (py as u32) < h
                    && edges.get_pixel(px as u32, py as u32)[0] > 0
                {
                    return true;
                }
            }
        }
        false
    };
    let mut out = Vec::new();
    for l in lines {
        let (a, b, c) = line_coeffs(l);
        // A point on the line, and its direction.
        let (px, py) = (-a * c, -b * c);
        let (dx, dy) = (-b, a);
        // March across the image extent in both directions.
        let span = ((w * w + h * h) as f64).sqrt();
        let (mut best_start, mut best_end, mut best_len) = (0.0f64, 0.0f64, 0.0f64);
        let (mut run_start, mut in_run, mut gap) = (0.0f64, false, 0i32);
        let mut t = -span;
        while t <= span {
            let (x, y) = (px + dx * t, py + dy * t);
            let inside = x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64;
            let hit = inside && is_edge(x, y);
            if hit {
                if !in_run {
                    run_start = t;
                    in_run = true;
                }
                gap = 0;
            } else if in_run {
                gap += 1;
                // Tolerate short breaks so a dashed or partly-occluded edge
                // still reads as one segment.
                if gap > 6 {
                    let len = t - gap as f64 - run_start;
                    if len > best_len {
                        best_len = len;
                        best_start = run_start;
                        best_end = t - gap as f64;
                    }
                    in_run = false;
                }
            }
            t += 1.0;
        }
        if in_run {
            let len = span - run_start;
            if len > best_len {
                best_len = len;
                best_start = run_start;
                best_end = span;
            }
        }
        if best_len >= min_len {
            out.push(Segment {
                a: (px + dx * best_start, py + dy * best_start),
                b: (px + dx * best_end, py + dy * best_end),
                line: (a, b, c),
            });
        }
    }
    out
}

/// Greedily extracts dominant vanishing points by RANSAC over line
/// intersections, removing each VP's supporting lines before seeking the next.
fn find_vanishing_points(
    segs: &[Segment],
    inlier_px: f64,
    want: usize,
) -> Vec<((f64, f64), usize)> {
    let mut remaining: Vec<usize> = (0..segs.len()).collect();
    let mut rng = Xorshift64(0x7A17_C0DE_1234_5678);
    let mut out = Vec::new();

    for _ in 0..want {
        if remaining.len() < 4 {
            break;
        }
        let mut best: Option<((f64, f64), Vec<usize>)> = None;
        for _ in 0..3000 {
            let i = remaining[rng.below(remaining.len())];
            let j = remaining[rng.below(remaining.len())];
            if i == j {
                continue;
            }
            let Some(vp) = intersect(segs[i].line, segs[j].line) else {
                continue;
            };
            if vp.0.abs() > 1e6 || vp.1.abs() > 1e6 {
                continue;
            }
            let support: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&k| segs[k].supports(vp, inlier_px))
                .collect();
            if best
                .as_ref()
                .map(|(_, s)| support.len() > s.len())
                .unwrap_or(true)
            {
                best = Some((vp, support));
            }
        }
        let Some((vp, support)) = best else { break };
        if support.len() < 4 {
            break;
        }
        out.push((vp, support.len()));
        remaining.retain(|k| !support.contains(k));
    }
    out
}

pub struct VanishingCalibration {
    pub focal_px: f64,
    /// How many independent perpendicular VP pairs agreed.
    pub num_pairs: usize,
}

/// Estimates focal length from one image's vanishing points.
pub fn estimate_focal(gray: &GrayImage) -> Option<VanishingCalibration> {
    let (w, h) = gray.dimensions();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);

    let edges = imageproc::edges::canny(gray, 40.0, 100.0);
    // Vote threshold scaled to image size: a fixed count would find nothing on
    // a small image and thousands of spurious lines on a large one. Tuned on
    // real photos - a quarter of the short side demanded near-full-height
    // straight edges and returned 5-8 lines per image, far too few for
    // vanishing points to emerge; 8% finds the structural edges that actually
    // carry the scene's orthogonality.
    let vote = ((w.min(h) as f32) * 0.08) as u32;
    let lines: Vec<PolarLine> = detect_lines(
        &edges,
        LineDetectionOptions {
            vote_threshold: vote.max(30),
            suppression_radius: 5,
        },
    );
    if lines.len() < 6 {
        return None;
    }
    // Only segments long enough to have a trustworthy direction.
    let min_len = (w.min(h) as f64) * 0.10;
    let segs = segments_from_edges(&edges, &lines, min_len);
    if segs.len() < 6 {
        return None;
    }

    let inlier_px = (w.max(h) as f64) * 0.004;
    let vps = find_vanishing_points(&segs, inlier_px.max(1.5), 3);
    if vps.len() < 2 {
        return None;
    }

    // Every VP pair that yields a positive f^2 is a perpendicular pair; pairs
    // that don't simply aren't orthogonal directions, which is information
    // rather than an error.
    let mut focals: Vec<f64> = Vec::new();
    for a in 0..vps.len() {
        for b in (a + 1)..vps.len() {
            let (v1, v2) = (vps[a].0, vps[b].0);
            let dot = (v1.0 - cx) * (v2.0 - cx) + (v1.1 - cy) * (v2.1 - cy);
            let f2 = -dot;
            if f2.is_finite() && f2 > 0.0 {
                let f = f2.sqrt();
                // Reject physically implausible focals outright: below ~0.2x
                // or above ~10x the image's long side is not a real lens, and
                // is a sign the "orthogonal" pair wasn't.
                let long = w.max(h) as f64;
                if f > 0.2 * long && f < 10.0 * long {
                    focals.push(f);
                }
            }
        }
    }
    if focals.is_empty() {
        return None;
    }
    focals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(VanishingCalibration {
        focal_px: focals[focals.len() / 2],
        num_pairs: focals.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_vanishing_points_give_the_focal() {
        // Two perpendicular directions imaged by a camera with a known focal:
        // their vanishing points must satisfy the orthogonality relation.
        let (f, cx, cy) = (900.0f64, 640.0, 480.0);
        let d1 = [1.0f64, 0.0, 0.3];
        let d2 = [0.0f64, 1.0, 0.0];
        assert!((d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]).abs() < 1e-12);
        let vp = |d: [f64; 3]| (f * d[0] / d[2] + cx, f * d[1] / d[2] + cy);
        // d2 has zero z: its vanishing point is at infinity, so use a third
        // direction perpendicular to d1 with non-zero depth instead.
        let d3 = [-0.3f64, 0.0, 1.0];
        assert!((d1[0] * d3[0] + d1[1] * d3[1] + d1[2] * d3[2]).abs() < 1e-12);
        let (v1, v3) = (vp(d1), vp(d3));
        let dot = (v1.0 - cx) * (v3.0 - cx) + (v1.1 - cy) * (v3.1 - cy);
        let recovered = (-dot).sqrt();
        assert!(
            (recovered - f).abs() < 1e-6,
            "recovered {recovered}, want {f}"
        );
    }

    #[test]
    fn blank_image_abstains() {
        let img = GrayImage::from_pixel(200, 200, image::Luma([128]));
        assert!(estimate_focal(&img).is_none());
    }
}
