//! 4-point planar homography (DLT, `h33` fixed to 1). Used only to unwarp a
//! detected marker quad into its canonical square for bit sampling - not a
//! general-purpose N-point/robust homography estimator (that belongs in
//! `sfm-geometry` for two-view geometry, which uses RANSAC over many points).

use nalgebra::{SMatrix, SVector};

/// Solve for `h0..h7` (with `h8 = 1`) mapping `src[i] -> dst[i]` for all 4
/// correspondences. Returns `None` if the 4 points are degenerate (collinear
/// or duplicated), which should never happen for a real detected quad.
pub fn solve_homography(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<[f64; 8]> {
    let mut a = SMatrix::<f64, 8, 8>::zeros();
    let mut b = SVector::<f64, 8>::zeros();
    for i in 0..4 {
        let (x, y) = src[i];
        let (u, v) = dst[i];
        let row0 = 2 * i;
        let row1 = 2 * i + 1;
        a[(row0, 0)] = x;
        a[(row0, 1)] = y;
        a[(row0, 2)] = 1.0;
        a[(row0, 6)] = -x * u;
        a[(row0, 7)] = -y * u;
        b[row0] = u;
        a[(row1, 3)] = x;
        a[(row1, 4)] = y;
        a[(row1, 5)] = 1.0;
        a[(row1, 6)] = -x * v;
        a[(row1, 7)] = -y * v;
        b[row1] = v;
    }
    let h = a.lu().solve(&b)?;
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]])
}

pub fn apply_homography(h: &[f64; 8], x: f64, y: f64) -> (f64, f64) {
    let denom = h[6] * x + h[7] * y + 1.0;
    let u = (h[0] * x + h[1] * y + h[2]) / denom;
    let v = (h[3] * x + h[4] * y + h[5]) / denom;
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_square_maps_to_itself() {
        let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let h = solve_homography(src, src).unwrap();
        for &(x, y) in &src {
            let (u, v) = apply_homography(&h, x, y);
            assert!((u - x).abs() < 1e-9 && (v - y).abs() < 1e-9);
        }
    }

    #[test]
    fn maps_unit_square_to_arbitrary_quad() {
        let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst = [(10.0, 20.0), (110.0, 15.0), (105.0, 115.0), (8.0, 120.0)];
        let h = solve_homography(src, dst).unwrap();
        for i in 0..4 {
            let (u, v) = apply_homography(&h, src[i].0, src[i].1);
            assert!(
                (u - dst[i].0).abs() < 1e-6,
                "u mismatch at {i}: {u} vs {}",
                dst[i].0
            );
            assert!(
                (v - dst[i].1).abs() < 1e-6,
                "v mismatch at {i}: {v} vs {}",
                dst[i].1
            );
        }
        // Center of the unit square should land near the quad's centroid.
        let (cu, cv) = apply_homography(&h, 0.5, 0.5);
        let centroid_u: f64 = dst.iter().map(|p| p.0).sum::<f64>() / 4.0;
        let centroid_v: f64 = dst.iter().map(|p| p.1).sum::<f64>() / 4.0;
        assert!((cu - centroid_u).abs() < 10.0);
        assert!((cv - centroid_v).abs() < 10.0);
    }
}
