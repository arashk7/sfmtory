//! Square fiducial marker ("ArUco-style") detector: adaptive threshold,
//! connected-component blob extraction, convex-hull + rotating-calipers quad
//! fitting, homography-based bit sampling, and matching against a small
//! generated dictionary.
//!
//! This is **not** byte-compatible with OpenCV's standard ArUco dictionaries
//! (`DICT_4X4_50` etc.) - it uses its own dictionary of 4x4-bit codes,
//! generated once at startup with a minimum pairwise (and self-rotation)
//! Hamming distance for reliable identification. For this project's use case
//! (rigid, printable markers as very high-confidence correspondences for
//! calibration) that's sufficient; markers must be printed from
//! `sfm::aruco::dictionary()` (or `sfm features print-markers`, once that CLI
//! command exists) rather than from an OpenCV-generated sheet.
//!
//! Each detected marker contributes 4 keypoints (its corners) with
//! `Descriptors::MarkerCorner` - matched across images by exact
//! `(marker_id, corner_index)` equality in `sfm-match`, not nearest-neighbor
//! search, since the ID makes correspondence unambiguous.

use std::sync::OnceLock;

use sfm_core::{Descriptors, FeatureSet, Keypoint};

use crate::geom2d::{convex_hull, min_area_rect, polygon_area};
use crate::homography::{apply_homography, solve_homography};

const DATA_BITS: usize = 4;
const GRID: usize = DATA_BITS + 2; // +1 black border cell on each side

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ArucoParams {
    /// Half-size (pixels) of the local-mean window for adaptive thresholding.
    pub adaptive_radius: i32,
    /// A pixel is "ink" when it's this much darker than its local mean.
    pub adaptive_c: f32,
    pub min_component_pixels: usize,
    pub min_perimeter_px: f64,
    pub max_hamming_distance: u32,
    pub num_dictionary_markers: usize,
    /// Contrast gain applied to the greyscale image before thresholding,
    /// about mid-grey. Values above 1 pull a flat, low-contrast capture apart
    /// so the adaptive threshold has an edge to find; 1.0 is a no-op.
    pub contrast: f32,
    /// Gamma applied before thresholding. Below 1 lifts shadows (markers lost
    /// in a dark frame), above 1 pulls down highlights (markers washed out by
    /// over-exposure); 1.0 is a no-op.
    pub gamma: f32,
}

impl Default for ArucoParams {
    fn default() -> Self {
        ArucoParams {
            adaptive_radius: 7,
            adaptive_c: 12.0,
            min_component_pixels: 80,
            min_perimeter_px: 60.0,
            max_hamming_distance: 3,
            num_dictionary_markers: 50,
            contrast: 1.0,
            gamma: 1.0,
        }
    }
}

pub fn dictionary(num_markers: usize) -> &'static Vec<u16> {
    static DICT: OnceLock<Vec<u16>> = OnceLock::new();
    DICT.get_or_init(|| generate_dictionary(num_markers, 4))
}

fn rotate90(bits: u16) -> u16 {
    let mut out = 0u16;
    for row in 0..4usize {
        for col in 0..4usize {
            let bit = (bits >> (row * 4 + col)) & 1;
            let (nrow, ncol) = (col, 3 - row);
            out |= bit << (nrow * 4 + ncol);
        }
    }
    out
}

fn hamming(a: u16, b: u16) -> u32 {
    (a ^ b).count_ones()
}

fn min_rotated_hamming(a: u16, b: u16) -> u32 {
    let mut best = hamming(a, b);
    let mut r = a;
    for _ in 0..3 {
        r = rotate90(r);
        best = best.min(hamming(r, b));
    }
    best
}

fn generate_dictionary(num: usize, min_dist: u32) -> Vec<u16> {
    let mut codes: Vec<u16> = Vec::new();
    for c in 0..=0xFFFFu32 {
        let c = c as u16;
        let ones = c.count_ones();
        if ones < 3 || ones > 13 {
            continue;
        }
        let r1 = rotate90(c);
        let r2 = rotate90(r1);
        let r3 = rotate90(r2);
        if hamming(c, r1) == 0 || hamming(c, r2) == 0 || hamming(c, r3) == 0 {
            continue;
        }
        if codes
            .iter()
            .any(|&existing| min_rotated_hamming(c, existing) < min_dist)
        {
            continue;
        }
        codes.push(c);
        if codes.len() >= num {
            break;
        }
    }
    codes
}

struct IntegralImage {
    sums: Vec<u64>,
    w: usize,
    h: usize,
}

impl IntegralImage {
    fn build(gray: &image::GrayImage) -> Self {
        let (w, h) = gray.dimensions();
        let (w, h) = (w as usize, h as usize);
        // Straight off the raw buffer: `get_pixel` is a bounds-checked call per
        // pixel, and at 12 megapixels that overhead is the whole cost of an
        // otherwise trivial prefix sum.
        let src = gray.as_raw();
        let stride = w + 1;
        let mut sums = vec![0u64; stride * (h + 1)];
        for y in 0..h {
            let row = &src[y * w..y * w + w];
            let (prev, cur) = sums.split_at_mut((y + 1) * stride);
            let prev = &prev[y * stride..];
            let mut row_sum = 0u64;
            for x in 0..w {
                row_sum += row[x] as u64;
                cur[x + 1] = prev[x + 1] + row_sum;
            }
        }
        IntegralImage { sums, w, h }
    }

    /// Mean intensity over `[x0,x1) x [y0,y1)`, clamped to image bounds.
    fn mean(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> f32 {
        let x0 = x0.clamp(0, self.w as i32) as usize;
        let y0 = y0.clamp(0, self.h as i32) as usize;
        let x1 = x1.clamp(0, self.w as i32) as usize;
        let y1 = y1.clamp(0, self.h as i32) as usize;
        if x1 <= x0 || y1 <= y0 {
            return 255.0;
        }
        let stride = self.w + 1;
        // Sum first, subtract once: `a - b - c + d` can underflow at the
        // intermediate `a - b` step even though the final box sum is always
        // non-negative (the two negative terms can each individually exceed
        // the corner term on their own).
        let positive = self.sums[y1 * stride + x1] + self.sums[y0 * stride + x0];
        let negative = self.sums[y0 * stride + x1] + self.sums[y1 * stride + x0];
        let sum = positive - negative;
        sum as f32 / ((x1 - x0) * (y1 - y0)) as f32
    }
}

fn adaptive_ink_mask(gray: &image::GrayImage, params: &ArucoParams) -> Vec<bool> {
    use rayon::prelude::*;
    let (w, h) = gray.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let integral = IntegralImage::build(gray);
    let r = params.adaptive_radius;
    let c = params.adaptive_c;
    let src = gray.as_raw();
    let mut mask = vec![false; wu * hu];
    // Rows are independent given the integral image, and this is pure
    // arithmetic over 12M pixels - the one stage in the detector that
    // parallelises for free.
    mask.par_chunks_mut(wu).enumerate().for_each(|(y, row)| {
        let yi = y as i32;
        let vals = &src[y * wu..y * wu + wu];
        for (x, m) in row.iter_mut().enumerate() {
            let xi = x as i32;
            let mean = integral.mean(xi - r, yi - r, xi + r + 1, yi + r + 1);
            *m = vals[x] as f32 + c < mean;
        }
    });
    mask
}

fn label_components(mask: &[bool], w: usize, h: usize) -> Vec<i32> {
    let mut labels = vec![-1i32; w * h];
    let mut next_label = 0i32;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || labels[start] != -1 {
            continue;
        }
        labels[start] = next_label;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            let x = idx % w;
            let y = idx / w;
            let neighbors = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= w || ny >= h {
                    continue;
                }
                let nidx = ny * w + nx;
                if mask[nidx] && labels[nidx] == -1 {
                    labels[nidx] = next_label;
                    stack.push(nidx);
                }
            }
        }
        next_label += 1;
    }
    labels
}

/// Border pixels of every kept component, grouped by label.
///
/// Stored CSR-style - one flat point array plus per-label offsets - rather
/// than a `Vec<Vec<_>>`, so a 12MP image with 11k components does not perform
/// 11k separate allocations to hold half a million points.
struct ComponentBorders {
    points: Vec<(f64, f64)>,
    /// Label `l` owns `points[offsets[l]..offsets[l + 1]]`.
    offsets: Vec<u32>,
}

impl ComponentBorders {
    fn get(&self, label: usize) -> &[(f64, f64)] {
        &self.points[self.offsets[label] as usize..self.offsets[label + 1] as usize]
    }
}

/// Collects every component's border pixels in two passes over the image.
///
/// This replaces a per-label full-image scan, which made border extraction
/// O(width x height x labels) and completely dominated detection: on a real
/// 3840x3104 frame with 1275 components above the size floor, it was 10.63s of
/// a 12.28s detect() - 15 billion pixel visits to collect 498k points. Two
/// passes over the image cost O(width x height) regardless of how many
/// components there are.
fn extract_borders(labels: &[i32], w: usize, h: usize, keep: &[bool]) -> ComponentBorders {
    let num_labels = keep.len();
    // Pass 1: how many border pixels each kept component has.
    let mut offsets = vec![0u32; num_labels + 1];
    let visit = |f: &mut dyn FnMut(usize, usize, usize)| {
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let l = labels[row + x];
                if l < 0 {
                    continue;
                }
                let l = l as usize;
                if !keep[l] {
                    continue;
                }
                let is_border = x == 0
                    || y == 0
                    || x == w - 1
                    || y == h - 1
                    || labels[row + x - 1] != l as i32
                    || labels[row + x + 1] != l as i32
                    || labels[row - w + x] != l as i32
                    || labels[row + w + x] != l as i32;
                if is_border {
                    f(l, x, y);
                }
            }
        }
    };
    visit(&mut |l, _, _| offsets[l + 1] += 1);

    // Prefix sum turns the counts into start offsets.
    for i in 0..num_labels {
        offsets[i + 1] += offsets[i];
    }

    // Pass 2: place each point, using a moving cursor per label.
    let total = offsets[num_labels] as usize;
    let mut points = vec![(0.0, 0.0); total];
    let mut cursor = offsets.clone();
    visit(&mut |l, x, y| {
        points[cursor[l] as usize] = (x as f64, y as f64);
        cursor[l] += 1;
    });

    ComponentBorders { points, offsets }
}

fn order_by_angle_around_centroid(mut corners: [(f64, f64); 4]) -> [(f64, f64); 4] {
    let cx = corners.iter().map(|p| p.0).sum::<f64>() / 4.0;
    let cy = corners.iter().map(|p| p.1).sum::<f64>() / 4.0;
    corners.sort_by(|a, b| {
        let angle_a = (a.1 - cy).atan2(a.0 - cx);
        let angle_b = (b.1 - cy).atan2(b.0 - cx);
        angle_a.partial_cmp(&angle_b).unwrap()
    });
    corners
}

/// Applies the exposure-compensating preprocessing (`contrast`, `gamma`)
/// before thresholding. A no-op at the defaults, so datasets that don't need
/// it pay nothing.
fn preprocess(gray: &image::GrayImage, params: &ArucoParams) -> image::GrayImage {
    let mut out = gray.clone();
    for p in out.pixels_mut() {
        let mut v = p[0] as f32 / 255.0;
        if params.gamma != 1.0 {
            v = v.powf(params.gamma);
        }
        if params.contrast != 1.0 {
            v = (v - 0.5) * params.contrast + 0.5;
        }
        p[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

pub fn detect(img: &image::DynamicImage, params: &ArucoParams) -> FeatureSet {
    let luma = crate::gray::to_luma8_par(img);
    // Thresholding sees the exposure-corrected image; grid sampling deliberately
    // still sees the original, so `contrast`/`gamma` only ever help find the
    // quads and never alter the bits read out of one. At the defaults both are
    // the same buffer and nothing is copied.
    let gray_u8 = if params.contrast == 1.0 && params.gamma == 1.0 {
        std::borrow::Cow::Borrowed(&luma)
    } else {
        std::borrow::Cow::Owned(preprocess(&luma, params))
    };
    let (w, h) = gray_u8.dimensions();
    let (wu, hu) = (w as usize, h as usize);

    let mask = adaptive_ink_mask(&gray_u8, params);
    let labels = label_components(&mask, wu, hu);
    let num_labels = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);

    let mut counts = vec![0usize; num_labels.max(0) as usize];
    for &l in &labels {
        if l >= 0 {
            counts[l as usize] += 1;
        }
    }

    let dict = dictionary(params.num_dictionary_markers);
    let mut keypoints = Vec::new();
    let mut marker_data = Vec::new();

    let keep: Vec<bool> = counts
        .iter()
        .map(|c| *c >= params.min_component_pixels)
        .collect();
    let borders = extract_borders(&labels, wu, hu, &keep);

    for label in 0..num_labels {
        if !keep[label as usize] {
            continue;
        }
        let border = borders.get(label as usize);
        if border.len() < 8 {
            continue;
        }
        let hull = convex_hull(border);
        let rect = match min_area_rect(&hull) {
            Some(r) => r,
            None => continue,
        };
        let perimeter: f64 = (0..4)
            .map(|i| {
                let a = rect[i];
                let b = rect[(i + 1) % 4];
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum();
        if perimeter < params.min_perimeter_px {
            continue;
        }
        let hull_area = polygon_area(&hull);
        let rect_area = polygon_area(&rect);
        if rect_area < 1e-6 || hull_area / rect_area < 0.75 {
            continue;
        }

        let dst = order_by_angle_around_centroid(rect);
        // Match src's traversal winding to dst's so the sampled grid isn't
        // mirrored (see module docs on rotation-only dictionary matching).
        let dst_signed_area = shoelace_signed(&dst);
        let src = if dst_signed_area >= 0.0 {
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        } else {
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        };
        let h_matrix = match solve_homography(src, dst) {
            Some(h) => h,
            None => continue,
        };

        let sampled = sample_grid_bits(&luma, &h_matrix);
        let border_ok = check_border_cells(sampled);
        if !border_ok {
            continue;
        }
        let data_bits = extract_data_bits(sampled);

        let mut best: Option<(u32, u32, u32)> = None; // (marker_id, rotation, distance)
        for (id, &code) in dict.iter().enumerate() {
            let mut c = code;
            for k in 0..4u32 {
                let d = hamming(c, data_bits);
                if best.is_none_or(|(_, _, bd)| d < bd) {
                    best = Some((id as u32, k, d));
                }
                c = rotate90(c);
            }
        }
        let (marker_id, rotation, distance) = match best {
            Some(v) if v.2 <= params.max_hamming_distance => v,
            _ => continue,
        };

        let mut ordered_corners = dst;
        ordered_corners.rotate_left(rotation as usize);

        for (corner_index, &(cx, cy)) in ordered_corners.iter().enumerate() {
            keypoints.push(Keypoint {
                x: cx as f32,
                y: cy as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0 / (1.0 + distance as f32),
            });
            // capture_id is stamped in later by the caller, which is the
            // only layer that knows the dataset layout - see
            // `stamp_capture_id`.
            marker_data.extend_from_slice(&0u32.to_le_bytes());
            marker_data.extend_from_slice(&marker_id.to_le_bytes());
            marker_data.extend_from_slice(&(corner_index as u32).to_le_bytes());
        }
    }

    FeatureSet {
        keypoints,
        descriptors: Descriptors::MarkerCorner { data: marker_data },
    }
}

fn shoelace_signed(pts: &[(f64, f64); 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % 4];
        sum += x1 * y2 - x2 * y1;
    }
    sum * 0.5
}

/// Sample every cell of the `GRID x GRID` canonical marker as a boolean
/// (`true` = ink/black), averaging a small window per cell for robustness.
/// Bilinear sample from the 8-bit grey image, normalised to `[0, 1]`.
///
/// Exists so the detector never materialises an `f32` copy of the frame. Grid
/// sampling touches a few tens of thousands of points across a whole image;
/// converting 12 million pixels to `f32` to serve them cost 123ms and 47MB per
/// frame, for arithmetic identical to reading the `u8` and dividing.
fn sample_bilinear_u8(gray: &image::GrayImage, x: f32, y: f32) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let src = gray.as_raw();
    let (w, y0, y1) = (w as usize, y0 as usize, y1 as usize);
    let at = |yy: usize, xx: u32| src[yy * w + xx as usize] as f32;
    let v0 = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
    let v1 = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
    (v0 * (1.0 - fy) + v1 * fy) / 255.0
}

fn sample_grid_bits(gray: &image::GrayImage, h: &[f64; 8]) -> [[bool; GRID]; GRID] {
    let mut cell_mean = [[0f32; GRID]; GRID];
    for r in 0..GRID {
        for c in 0..GRID {
            let mut sum = 0f32;
            let mut n = 0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let cx = (c as f64 + (sx as f64 + 1.0) / 4.0) / GRID as f64;
                    let cy = (r as f64 + (sy as f64 + 1.0) / 4.0) / GRID as f64;
                    let (ix, iy) = apply_homography(h, cx, cy);
                    sum += sample_bilinear_u8(gray, ix as f32, iy as f32);
                    n += 1;
                }
            }
            cell_mean[r][c] = sum / n as f32;
        }
    }
    let overall_mean: f32 = cell_mean.iter().flatten().sum::<f32>() / (GRID * GRID) as f32;
    let mut bits = [[false; GRID]; GRID];
    for r in 0..GRID {
        for c in 0..GRID {
            bits[r][c] = cell_mean[r][c] < overall_mean;
        }
    }
    bits
}

fn check_border_cells(bits: [[bool; GRID]; GRID]) -> bool {
    let mut ink = 0;
    let mut total = 0;
    for r in 0..GRID {
        for c in 0..GRID {
            let on_border = r == 0 || c == 0 || r == GRID - 1 || c == GRID - 1;
            if on_border {
                total += 1;
                if bits[r][c] {
                    ink += 1;
                }
            }
        }
    }
    (ink as f32 / total as f32) >= 0.8
}

fn extract_data_bits(bits: [[bool; GRID]; GRID]) -> u16 {
    let mut out = 0u16;
    for r in 0..DATA_BITS {
        for c in 0..DATA_BITS {
            if bits[r + 1][c + 1] {
                out |= 1 << (r * DATA_BITS + c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    /// Render one dictionary marker as a synthetic image: white background,
    /// a black `GRID x GRID` marker (each cell `cell_px` pixels) at `(ox, oy)`.
    fn render_marker(
        marker_id: usize,
        cell_px: u32,
        ox: u32,
        oy: u32,
        canvas: u32,
    ) -> DynamicImage {
        let code = dictionary(50)[marker_id];
        let bits_at = |r: usize, c: usize| -> bool {
            if r == 0 || c == 0 || r == GRID - 1 || c == GRID - 1 {
                true
            } else {
                (code >> ((r - 1) * DATA_BITS + (c - 1))) & 1 == 1
            }
        };
        let img = GrayImage::from_fn(canvas, canvas, |x, y| {
            if x < ox || y < oy {
                return Luma([255]);
            }
            let gx = (x - ox) / cell_px;
            let gy = (y - oy) / cell_px;
            if gx as usize >= GRID || gy as usize >= GRID {
                return Luma([255]);
            }
            if bits_at(gy as usize, gx as usize) {
                Luma([20])
            } else {
                Luma([235])
            }
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn detects_a_rendered_marker_and_recovers_its_id() {
        let marker_id = 3;
        let img = render_marker(marker_id, 20, 40, 40, 300);
        let features = detect(&img, &ArucoParams::default());
        assert_eq!(
            features.len(),
            4,
            "expected exactly 4 corner keypoints for one marker"
        );
        for i in 0..4 {
            let (_capture, id, _corner) = features.descriptors.marker_corner(i).unwrap();
            assert_eq!(id as usize, marker_id);
        }
    }

    #[test]
    fn finds_nothing_on_blank_image() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(200, 200, Luma([255])));
        let features = detect(&img, &ArucoParams::default());
        assert!(features.is_empty());
    }

    #[test]
    fn dictionary_codes_are_pairwise_distinct_under_rotation() {
        let dict = dictionary(50);
        for i in 0..dict.len() {
            for j in (i + 1)..dict.len() {
                assert!(min_rotated_hamming(dict[i], dict[j]) >= 4);
            }
        }
    }
}

/// Rewrites every marker-corner descriptor's `capture_id` field in place.
///
/// The detector sees one image and has no idea which capture it belongs to;
/// the dataset layout does. Keeping the stamping separate lets the detector
/// stay layout-agnostic while still producing descriptors whose identity is
/// correct across captures (see `sfm_core::Descriptors::MarkerCorner`).
pub fn stamp_capture_id(features: &mut sfm_core::FeatureSet, capture_id: u32) {
    if let sfm_core::Descriptors::MarkerCorner { data } = &mut features.descriptors {
        for row in data.chunks_exact_mut(sfm_core::MARKER_CORNER_BYTES) {
            row[0..4].copy_from_slice(&capture_id.to_le_bytes());
        }
    }
}
