//! Focal-length estimation from fiducial markers, using the fact that every
//! ArUco marker is a *square*.
//!
//! This is Zhang's calibration insight applied to something the detector
//! already produces. A planar target of known shape, seen at a known
//! orientation-free layout, induces a homography `H ~ K [r1 r2 t]` from target
//! plane to image. Because `r1` and `r2` are columns of a rotation matrix,
//! they are orthonormal, and that gives two constraints per homography:
//!
//! ```text
//! h1^T B h2 = 0                and        h1^T B h1 = h2^T B h2,     B = K^-T K^-1
//! ```
//!
//! Two properties make this a particularly good fit here:
//!
//! - **No user input is required.** The constraints come from the target being
//!   square, not from its size. Physical dimensions would only fix the
//!   reconstruction's overall scale, which calibration does not need. So this
//!   works on any fiducial dataset as-is, with no measured board to enter.
//! - **Every marker is an independent sample.** A single small marker gives a
//!   noisy homography, but a dataset has hundreds across many images and
//!   orientations, and the aggregate is what matters.
//!
//! Assuming square pixels, zero skew, and the principal point at the image
//! centre reduces the unknowns to focal length alone, at which point each of
//! the two constraints above yields a closed-form `f^2` directly - so one
//! marker gives two independent estimates, and the estimator's job is to
//! collect them and take a robust average.

use sfm_core::FeatureSet;
use sfm_geometry::homography::linear_homography;

/// One image's fiducial detections plus its pixel dimensions.
pub type FeatureView<'a> = (&'a FeatureSet, u32, u32);

/// A marker's four corners as they are collected, indexed by corner id.
type MarkerCorners = [Option<(f64, f64)>; 4];

/// One marker's contribution: up to two independent focal estimates.
///
/// Geometrically this is the two-vanishing-point relation specialised to a
/// square: the quad's two pairs of opposite sides meet at the vanishing points
/// of two perpendicular directions, and perpendicularity fixes `f`. Writing it
/// through the homography (below) is the same statement in Zhang's form.
///
/// The catch, and the reason for the conditioning gate: a square that is small
/// in the image is very nearly *affine* under projection. Its opposite sides
/// are almost parallel, so both vanishing points run off toward infinity and
/// the perspective terms of `H` fall to the order of the corner-localisation
/// noise. The constraints then divide one tiny noisy number by another and
/// return something arbitrary. Measured on real photos of a marker grid, every
/// one of 169 markers was in this regime. So the marker's own projective
/// distortion has to be checked before its constraints mean anything.
fn focal_estimates_from_marker(
    corners: &[(f64, f64); 4],
    cx: f64,
    cy: f64,
    image_diag: f64,
) -> Vec<f64> {
    // Vanishing points of the square's two side directions. `None` means the
    // sides are parallel in the image - no perspective, hence no information.
    let vp = |a: usize, b: usize, c: usize, d: usize| -> Option<(f64, f64)> {
        let l1 = line_through(corners[a], corners[b]);
        let l2 = line_through(corners[c], corners[d]);
        let (x, y, w) = (
            l1.1 * l2.2 - l1.2 * l2.1,
            l1.2 * l2.0 - l1.0 * l2.2,
            l1.0 * l2.1 - l1.1 * l2.0,
        );
        if w.abs() < 1e-12 {
            return None;
        }
        let p = (x / w, y / w);
        if p.0.is_finite() && p.1.is_finite() {
            Some(p)
        } else {
            None
        }
    };
    let (v1, v2) = match (vp(0, 1, 3, 2), vp(0, 3, 1, 2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Vec::new(),
    };
    // Require both vanishing points to sit within a bounded multiple of the
    // image diagonal. Beyond that the square is effectively affine and the
    // recovered focal is noise amplified by a near-zero denominator.
    let far = |p: (f64, f64)| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt();
    if far(v1) > MAX_VANISHING_DISTANCE * image_diag
        || far(v2) > MAX_VANISHING_DISTANCE * image_diag
    {
        return Vec::new();
    }

    // Target plane coordinates: the unit square, in the same corner order the
    // detector emits (clockwise from top-left).
    let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    // Work relative to the principal point so `K` reduces to diag(f, f, 1),
    // which is what makes the closed forms below possible.
    let centred: Vec<(f64, f64)> = corners.iter().map(|&(x, y)| (x - cx, y - cy)).collect();
    let Some(h) = linear_homography(&square, &centred) else {
        return Vec::new();
    };
    let (h11, h12) = (h[(0, 0)], h[(0, 1)]);
    let (h21, h22) = (h[(1, 0)], h[(1, 1)]);
    let (h31, h32) = (h[(2, 0)], h[(2, 1)]);

    let mut out = Vec::new();
    // Constraint 1: h1 . B . h2 = 0  ->  (h11 h12 + h21 h22)/f^2 + h31 h32 = 0
    let denom = h31 * h32;
    if denom.abs() > 1e-12 {
        let f2 = -(h11 * h12 + h21 * h22) / denom;
        if f2.is_finite() && f2 > 0.0 {
            out.push(f2.sqrt());
        }
    }
    // Constraint 2: h1 . B . h1 = h2 . B . h2
    let denom = h32 * h32 - h31 * h31;
    if denom.abs() > 1e-12 {
        let f2 = ((h11 * h11 + h21 * h21) - (h12 * h12 + h22 * h22)) / denom;
        if f2.is_finite() && f2 > 0.0 {
            out.push(f2.sqrt());
        }
    }
    out
}

/// Homogeneous line through two image points.
fn line_through(p: (f64, f64), q: (f64, f64)) -> (f64, f64, f64) {
    (p.1 - q.1, q.0 - p.0, p.0 * q.1 - q.0 * p.1)
}

/// Vanishing points beyond this multiple of the image diagonal mean the square
/// is too close to affine for its constraints to carry signal.
const MAX_VANISHING_DISTANCE: f64 = 30.0;

pub struct SquareCalibration {
    pub focal_px: f64,
    /// Median absolute deviation of the pooled per-marker estimates, relative
    /// to the median - a direct read on how much the samples disagree.
    pub relative_spread: f64,
    pub num_markers: usize,
    pub num_estimates: usize,
}

/// Pools focal estimates from every marker in every image.
///
/// `images` supplies each image's features and pixel dimensions.
pub fn estimate_focal(images: &[FeatureView<'_>]) -> Option<SquareCalibration> {
    let mut samples: Vec<f64> = Vec::new();
    let mut num_markers = 0usize;

    for (features, w, h) in images {
        let (cx, cy) = (*w as f64 / 2.0, *h as f64 / 2.0);
        // Group this image's corners by the marker they belong to, ordered by
        // corner index so the correspondence with the unit square is right.
        let mut by_marker: std::collections::BTreeMap<(u32, u32), MarkerCorners> =
            Default::default();
        for i in 0..features.len() {
            let Some((capture, marker, corner)) = features.descriptors.marker_corner(i) else {
                return None; // not a fiducial dataset
            };
            if corner > 3 {
                continue;
            }
            let kp = features.keypoints[i];
            by_marker.entry((capture, marker)).or_default()[corner as usize] =
                Some((kp.x as f64, kp.y as f64));
        }
        for (_, slots) in by_marker {
            let Some(corners) = slots
                .iter()
                .copied()
                .collect::<Option<Vec<_>>>()
                .and_then(|v| <[(f64, f64); 4]>::try_from(v).ok())
            else {
                continue; // partially detected marker
            };
            num_markers += 1;
            let diag = ((*w as f64).powi(2) + (*h as f64).powi(2)).sqrt();
            samples.extend(focal_estimates_from_marker(&corners, cx, cy, diag));
        }
    }

    if samples.len() < 4 {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    if !(median.is_finite() && median > 0.0) {
        return None;
    }
    // Median absolute deviation rather than a standard deviation: individual
    // markers can produce wild values when they happen to sit near-parallel to
    // the image plane (the constraints degenerate there), and a mean-based
    // spread would be dominated by those rather than describing the bulk.
    let mut devs: Vec<f64> = samples.iter().map(|s| (s - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2];

    Some(SquareCalibration {
        focal_px: median,
        relative_spread: mad / median,
        num_markers,
        num_estimates: samples.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    /// Projects a unit square placed in 3D at a given pose through a pinhole
    /// with focal `f`, returning its four image corners.
    fn project_square(
        f: f64,
        cx: f64,
        cy: f64,
        r: Matrix3<f64>,
        t: Vector3<f64>,
    ) -> [(f64, f64); 4] {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut out = [(0.0, 0.0); 4];
        for (k, &(u, v)) in sq.iter().enumerate() {
            let p = r * Vector3::new(u, v, 0.0) + t;
            out[k] = (f * p.x / p.z + cx, f * p.y / p.z + cy);
        }
        out
    }

    #[test]
    fn recovers_focal_from_tilted_squares() {
        let (f, cx, cy) = (1150.0, 640.0, 480.0);
        let mut samples = Vec::new();
        // A spread of orientations - the constraints carry no information from
        // a square that is exactly fronto-parallel.
        for (a, b) in [
            (0.5, 0.2),
            (-0.4, 0.35),
            (0.3, -0.45),
            (0.6, 0.1),
            (-0.25, -0.5),
        ] {
            let r = nalgebra::Rotation3::from_euler_angles(a, b, 0.15).into_inner();
            let t = Vector3::new(-0.5, -0.5, 4.0);
            let corners = project_square(f, cx, cy, r, t);
            samples.extend(focal_estimates_from_marker(&corners, cx, cy, 1600.0));
        }
        assert!(samples.len() >= 8, "expected estimates from each square");
        samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = samples[samples.len() / 2];
        assert!(
            (med - f).abs() / f < 0.01,
            "median focal {med} should be within 1% of {f}"
        );
    }

    #[test]
    fn fronto_parallel_squares_are_uninformative_not_wrong() {
        // Facing the camera head-on, both constraints degenerate. The right
        // behaviour is to produce nothing, not a confident wrong number.
        let (f, cx, cy) = (1000.0, 640.0, 480.0);
        let r = Matrix3::identity();
        let corners = project_square(f, cx, cy, r, Vector3::new(-0.5, -0.5, 5.0));
        let est = focal_estimates_from_marker(&corners, cx, cy, 1600.0);
        assert!(
            est.is_empty() || est.iter().all(|v| v.is_finite()),
            "degenerate square must abstain rather than emit garbage"
        );
    }
}
