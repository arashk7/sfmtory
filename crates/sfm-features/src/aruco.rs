//! Square fiducial marker ("ArUco-style") detector: adaptive threshold,
//! connected-component blob extraction, polygon approximation to four corners,
//! homography-based cell sampling, and identification from the marker's own
//! bit pattern.
//!
//! # Working with any family, without its table
//!
//! By default a marker's id **is** its pattern: the cells are read, reduced to
//! the smallest of their four rotations, and that canonical word is the id. No
//! dictionary is consulted, so a board printed from any square-fiducial family
//! (OpenCV's `DICT_4X4_1000`, an AprilTag sheet, anything) is identified
//! consistently, because the canonical form is a property of the ink rather
//! than of a numbering scheme. That is exactly what correspondence needs: the
//! same physical marker gets the same id in every frame, from every angle.
//!
//! What this deliberately does *not* do is recover the number printed beside
//! the marker. That requires the family's own table, which is a fixed data
//! blob per family and not derivable from an image. Matching a table would also
//! error-correct; identification here instead refuses to read a marker whose
//! cells are not cleanly separated (see `MIN_CELL_CRISPNESS`), which is the
//! more conservative trade when the family is unknown.
//!
//! `dictionary()` still provides this crate's own generated 4x4 codes with a
//! guaranteed minimum rotated Hamming distance, for markers printed from it;
//! set `ArucoParams::dictionary_free = false` to match against it and get
//! error correction. The two modes assign unrelated ids, so a project must not
//! mix them.
//!
//! # Which family?
//!
//! The number of data cells per side is measured per frame rather than assumed.
//! Reading a 4x4 marker on a 5x5 grid does not fail cleanly: it samples across
//! cell boundaries and yields a confident, wrong code. See `choose_family`.
//!
//! Each detected marker contributes 4 keypoints (its corners) with
//! `Descriptors::MarkerCorner` - matched across images by exact
//! `(marker_id, corner_index)` equality in `sfm-match`, not nearest-neighbor
//! search, since the ID makes correspondence unambiguous.

use std::sync::OnceLock;

use sfm_core::{Descriptors, FeatureSet, Keypoint};

use crate::geom2d::{convex_hull, quad_from_hull};
use crate::homography::{apply_homography, solve_homography};

/// A detected quadrilateral and the homography taking the unit square onto it.
type Quad = ([(f64, f64); 4], [f64; 8]);

/// Data cells per side for this crate's own generated dictionary.
const DATA_BITS: usize = 4;

/// Minimum Hamming distance between any two dictionary codes, including all
/// four rotations of each.
const DICT_MIN_DISTANCE: u32 = 4;

/// Errors a code with `DICT_MIN_DISTANCE` can actually correct: the usual
/// `floor((d - 1) / 2)`.
///
/// This is not a tuning knob, and treating it as one was a real defect. The
/// default was 3 against a minimum distance of 4 - beyond the code's
/// correction capability, and catastrophically so: a Hamming ball of radius 3
/// in 16 bits holds 697 patterns, and 50 markers times 4 rotations times 697
/// covers 2.1x the entire 65536-pattern space. Every possible bit pattern
/// matched some marker, so the dictionary check accepted anything the quad
/// stage handed it and assigned it a confident, wrong id.
const MAX_CORRECTABLE_ERRORS: u32 = (DICT_MIN_DISTANCE - 1) / 2;

/// How far apart the black and white cell groups must sit, as a fraction of a
/// marker's own contrast range, before its bits are trusted.
///
/// This is what makes grid-size detection work: reading a marker at the wrong
/// size samples across cell boundaries, so the two groups run together and the
/// gap collapses. It doubles as a rejection test for square things that are
/// not markers at all.
const MIN_CELL_CRISPNESS: f32 = 0.15;

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
    /// Data cells per side. `None` detects it from the image.
    ///
    /// A board can be 4x4, 5x5, 6x6 or 7x7 and nothing about the capture says
    /// which. Reading a 4x4 marker on a 5x5 grid does not fail cleanly - it
    /// samples across cell boundaries and yields a confident, wrong code - so
    /// the size is measured rather than assumed.
    #[serde(default)]
    pub data_bits: Option<usize>,
    /// Identify markers by their own bit pattern instead of by matching this
    /// crate's generated dictionary.
    ///
    /// Defaults to `true`, which is what makes the detector work with a board
    /// printed from *any* square-fiducial family: the canonical rotation of the
    /// pattern is stable per physical marker, which is the whole of what
    /// correspondence needs. Set `false` only for markers printed from
    /// `dictionary()`, where matching the table also error-corrects.
    #[serde(default = "default_true")]
    pub dictionary_free: bool,
}

fn default_true() -> bool {
    true
}

/// Data-grid sizes considered when `data_bits` is not pinned.
pub const CANDIDATE_DATA_BITS: [usize; 4] = [4, 5, 6, 7];

impl Default for ArucoParams {
    fn default() -> Self {
        ArucoParams {
            adaptive_radius: 7,
            adaptive_c: 12.0,
            min_component_pixels: 80,
            min_perimeter_px: 60.0,
            max_hamming_distance: MAX_CORRECTABLE_ERRORS,
            num_dictionary_markers: 50,
            contrast: 1.0,
            gamma: 1.0,
            data_bits: None,
            dictionary_free: true,
        }
    }
}

pub fn dictionary(num_markers: usize) -> &'static Vec<u16> {
    static DICT: OnceLock<Vec<u16>> = OnceLock::new();
    DICT.get_or_init(|| generate_dictionary(num_markers, DICT_MIN_DISTANCE))
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
        if !(3..=13).contains(&ones) {
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

/// Picks the data-grid size the frame's markers actually use, with the number
/// of candidates that agreed.
///
/// Takes the *smallest* size that comes within `FAMILY_MARGIN` of the best
/// vote rather than the outright winner, because the error is one-sided.
/// Reading a 4x4 marker on a 7x7 grid can still land an all-ink outer ring -
/// the real border is wide enough to fill it - so oversized grids pick up
/// spurious votes and can even edge ahead. The reverse does not happen: read a
/// 7x7 marker as 4x4 and data cells intrude into the ring, which fails. So a
/// small grid's votes are trustworthy in a way a large grid's are not, and
/// preferring the smallest credible size is the conservative reading.
///
/// Measured on `scan2`: one frame voted 4x4:102 against 7x7:109 and was
/// decoded as 7x7, which is what produced duplicate ids within that frame -
/// the symptom that a family had been misread.
fn choose_family(
    luma: &image::GrayImage,
    candidates: &[Quad],
    params: &ArucoParams,
) -> (usize, usize) {
    /// A smaller grid wins unless a larger one beats it by more than this.
    const FAMILY_MARGIN: f64 = 1.4;

    let sizes: Vec<usize> = match params.data_bits {
        Some(n) => vec![n],
        None => CANDIDATE_DATA_BITS.to_vec(),
    };
    // Score by summed crispness, not by a count of markers that merely passed.
    // An oversized grid frequently still sees a solid ink ring - a marker's
    // border is thick enough that the ring lands inside it whatever the
    // spacing - so pass/fail alone barely separates the sizes. How cleanly the
    // cells split into black and white does: at the true size every sample sits
    // inside one cell, and at any other size some straddle a boundary.
    let tally: Vec<(usize, usize, f64)> = sizes
        .iter()
        .map(|&n| {
            let mut count = 0usize;
            let mut score = 0.0f64;
            for (_, h) in candidates {
                let cells = sample_cells(luma, h, n);
                if border_is_ink(&cells) && cells.crispness > MIN_CELL_CRISPNESS {
                    count += 1;
                    score += cells.crispness as f64;
                }
            }
            (n, count, score)
        })
        .collect();
    let best = tally.iter().map(|(_, _, s)| *s).fold(0.0f64, f64::max);
    let pick = tally
        .iter()
        .filter(|(_, c, s)| *c > 0 && *s * FAMILY_MARGIN >= best)
        .min_by_key(|(n, _, _)| *n)
        .copied();
    match pick {
        Some((n, c, _)) => (n, c),
        None => (*sizes.first().unwrap_or(&DATA_BITS), 0),
    }
}

/// Data cells per side that the frame's markers use, without decoding them.
///
/// Same vote `detect` runs, exposed so a tuning UI can report the family and a
/// caller can pin it afterwards. `None` when the frame carries no marker the
/// vote could agree on.
pub fn detect_family(img: &image::DynamicImage, params: &ArucoParams) -> Option<usize> {
    let luma = crate::gray::to_luma8_par(img);
    let gray = shade_corrected(&luma, params);
    let candidates = find_quads(gray.as_ref(), &luma, params);
    let (family, votes) = choose_family(&luma, &candidates, params);
    (votes > 0).then_some(family)
}

/// Exposure-corrected copy for thresholding, or the original when the
/// correction is a no-op.
fn shade_corrected<'a>(
    luma: &'a image::GrayImage,
    params: &ArucoParams,
) -> std::borrow::Cow<'a, image::GrayImage> {
    if params.contrast == 1.0 && params.gamma == 1.0 {
        std::borrow::Cow::Borrowed(luma)
    } else {
        std::borrow::Cow::Owned(preprocess(luma, params))
    }
}

/// Every quadrilateral in the frame big enough to be a marker, with the
/// homography taking the unit square onto it.
fn find_quads(
    gray_u8: &image::GrayImage,
    _luma: &image::GrayImage,
    params: &ArucoParams,
) -> Vec<Quad> {
    let (w, h) = gray_u8.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let mask = adaptive_ink_mask(gray_u8, params);
    let labels = label_components(&mask, wu, hu);
    let num_labels = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut counts = vec![0usize; num_labels.max(0) as usize];
    for &l in &labels {
        if l >= 0 {
            counts[l as usize] += 1;
        }
    }
    let keep: Vec<bool> = counts
        .iter()
        .map(|c| *c >= params.min_component_pixels)
        .collect();
    let borders = extract_borders(&labels, wu, hu, &keep);

    let mut out = Vec::new();
    for label in 0..num_labels {
        if !keep[label as usize] {
            continue;
        }
        let border = borders.get(label as usize);
        if border.len() < 8 {
            continue;
        }
        let hull = convex_hull(border);
        // The blob's actual four corners, not a bounding rectangle. See
        // `quad_from_hull` for why the previous minimum-area-rectangle fit was
        // wrong both as a filter and as a source of corner positions.
        let Some(quad) = quad_from_hull(&hull) else {
            continue;
        };
        let perimeter: f64 = (0..4)
            .map(|i| {
                let a = quad[i];
                let b = quad[(i + 1) % 4];
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum();
        if perimeter < params.min_perimeter_px {
            continue;
        }
        let dst = order_by_angle_around_centroid(quad);
        // Match src's traversal winding to dst's so the sampled grid isn't
        // mirrored (see module docs on rotation-only matching).
        let src = if shoelace_signed(&dst) >= 0.0 {
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        } else {
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        };
        if let Some(h) = solve_homography(src, dst) {
            out.push((dst, h));
        }
    }
    out
}

pub fn detect(img: &image::DynamicImage, params: &ArucoParams) -> FeatureSet {
    let luma = crate::gray::to_luma8_par(img);
    // Thresholding sees the exposure-corrected image; cell sampling deliberately
    // still sees the original, so `contrast`/`gamma` only ever help find the
    // quads and never alter the bits read out of one. At the defaults both are
    // the same buffer and nothing is copied.
    let gray_u8 = shade_corrected(&luma, params);

    let dict = dictionary(params.num_dictionary_markers);
    // A configured bound above the code's correction capability does not make
    // detection more tolerant, it makes identification meaningless, so it is
    // clamped rather than obeyed.
    let max_hamming = params.max_hamming_distance.min(MAX_CORRECTABLE_ERRORS);
    let mut keypoints = Vec::new();
    let mut marker_data = Vec::new();
    // Quads first, identity second: the family has to be decided across the
    // whole frame, so every candidate is collected before any is read.
    let candidates = find_quads(gray_u8.as_ref(), &luma, params);

    let (chosen, _) = choose_family(&luma, &candidates, params);

    for (dst, h_matrix) in candidates {
        let cells = sample_cells(&luma, &h_matrix, chosen);
        if !border_is_ink(&cells) || cells.crispness <= MIN_CELL_CRISPNESS {
            continue;
        }
        let word = data_word(&cells);

        let (marker_id, rotation, confidence) = if params.dictionary_free {
            // The pattern is its own name. See `canonical_word`.
            let (canonical, turns) = canonical_word(word, chosen);
            // All-ink or all-paper interiors are not markers; they are
            // whatever else in the scene happened to be square.
            let ones = canonical.count_ones() as usize;
            if ones == 0 || ones == chosen * chosen {
                continue;
            }
            (id_for(canonical, chosen), turns, cells.crispness)
        } else {
            if chosen != DATA_BITS {
                continue;
            }
            let mut best: Option<(u32, u32, u32)> = None;
            for (id, &code) in dict.iter().enumerate() {
                let mut c = code;
                for k in 0..4u32 {
                    let d = hamming(c, word as u16);
                    if best.is_none_or(|(_, _, bd)| d < bd) {
                        best = Some((id as u32, k, d));
                    }
                    c = rotate90(c);
                }
            }
            match best {
                Some((id, turns, d)) if d <= max_hamming => (id, turns, 1.0 / (1.0 + d as f32)),
                _ => continue,
            }
        };

        let mut ordered_corners = dst;
        ordered_corners.rotate_left(rotation as usize);

        for (corner_index, &(cx, cy)) in ordered_corners.iter().enumerate() {
            keypoints.push(Keypoint {
                x: cx as f32,
                y: cy as f32,
                scale: 1.0,
                angle: 0.0,
                response: confidence,
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

/// One marker's cells read off the image, for a given data-grid size.
struct Cells {
    /// `(data_bits + 2)` per side: the data grid plus its black border ring.
    grid: usize,
    /// `true` where the cell is ink.
    bits: Vec<bool>,
    /// How cleanly the cells separate into black and white, as the gap between
    /// the darkest white cell and the brightest black one over the full range.
    /// Negative when the two groups overlap, which means this grid size is
    /// slicing across cell boundaries rather than reading cells.
    crispness: f32,
}

/// Reads a marker's cells assuming `data_bits` data cells per side.
fn sample_cells(gray: &image::GrayImage, h: &[f64; 8], data_bits: usize) -> Cells {
    let grid = data_bits + 2;
    let mut cell_mean = vec![0f32; grid * grid];
    for r in 0..grid {
        for c in 0..grid {
            let mut sum = 0f32;
            let mut n = 0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let cx = (c as f64 + (sx as f64 + 1.0) / 4.0) / grid as f64;
                    let cy = (r as f64 + (sy as f64 + 1.0) / 4.0) / grid as f64;
                    let (ix, iy) = apply_homography(h, cx, cy);
                    sum += sample_bilinear_u8(gray, ix as f32, iy as f32);
                    n += 1;
                }
            }
            cell_mean[r * grid + c] = sum / n as f32;
        }
    }
    let overall: f32 = cell_mean.iter().sum::<f32>() / cell_mean.len() as f32;
    let bits: Vec<bool> = cell_mean.iter().map(|m| *m < overall).collect();

    // Separation between the two groups, normalised by the marker's own
    // contrast so it does not simply reward bright images.
    let (mut dark_hi, mut light_lo) = (f32::MIN, f32::MAX);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (m, b) in cell_mean.iter().zip(&bits) {
        lo = lo.min(*m);
        hi = hi.max(*m);
        if *b {
            dark_hi = dark_hi.max(*m);
        } else {
            light_lo = light_lo.min(*m);
        }
    }
    let range = (hi - lo).max(1e-6);
    let crispness = if dark_hi == f32::MIN || light_lo == f32::MAX {
        0.0
    } else {
        (light_lo - dark_hi) / range
    };
    Cells {
        grid,
        bits,
        crispness,
    }
}

/// The outermost ring must be entirely ink: that black border is what makes a
/// square fiducial identifiable at all, and it is the cheapest way to throw
/// out a quad that merely happens to be square.
fn border_is_ink(cells: &Cells) -> bool {
    let g = cells.grid;
    (0..g).all(|k| {
        cells.bits[k]
            && cells.bits[(g - 1) * g + k]
            && cells.bits[k * g]
            && cells.bits[k * g + g - 1]
    })
}

/// The interior cells as a bit word, row-major, ink = 1.
fn data_word(cells: &Cells) -> u64 {
    let g = cells.grid;
    let mut word = 0u64;
    for r in 1..g - 1 {
        for c in 1..g - 1 {
            word <<= 1;
            if cells.bits[r * g + c] {
                word |= 1;
            }
        }
    }
    word
}

/// Rotates an `n x n` bit word by 90 degrees clockwise.
fn rotate_word(word: u64, n: usize) -> u64 {
    let mut out = 0u64;
    for r in 0..n {
        for c in 0..n {
            // Destination (r, c) comes from source (n - 1 - c, r).
            let src = (n - 1 - c) * n + r;
            let bit = (word >> (n * n - 1 - src)) & 1;
            out |= bit << (n * n - 1 - (r * n + c));
        }
    }
    out
}

/// The smallest of a word's four rotations, and how many turns reach it.
///
/// This is what lets a marker be identified without any dictionary at all: the
/// canonical form is a property of the printed pattern, so the same physical
/// marker yields the same value in every image and from every angle, which is
/// exactly the correspondence the reconstruction needs. Matching a table would
/// additionally recover the *printed* id, but only for the one family whose
/// table you happen to hold.
fn canonical_word(word: u64, n: usize) -> (u64, u32) {
    let mut best = (word, 0u32);
    let mut w = word;
    for turn in 1..4u32 {
        w = rotate_word(w, n);
        if w < best.0 {
            best = (w, turn);
        }
    }
    best
}

/// A 32-bit id for a canonical code. Codes up to 32 bits (data grids to 5x5)
/// are their own id; larger ones are hashed, since the descriptor format
/// carries a `u32`.
fn id_for(canonical: u64, n: usize) -> u32 {
    if n * n <= 32 {
        canonical as u32
    } else {
        // FNV-1a, for a stable id across processes and runs.
        let mut h: u32 = 0x811c_9dc5;
        for byte in canonical.to_le_bytes() {
            h ^= byte as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    /// Render one dictionary marker as a synthetic image: white background,
    /// a black `GRID x GRID` marker (each cell `cell_px` pixels) at `(ox, oy)`.
    /// The generated dictionary's own layout: 4x4 data plus a border ring.
    const GRID: usize = DATA_BITS + 2;

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
        // Table matching is no longer the default, so ask for it explicitly.
        let params = ArucoParams {
            dictionary_free: false,
            data_bits: Some(DATA_BITS),
            ..Default::default()
        };
        let features = detect(&img, &params);
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

    /// Identification without any table: the id comes from the pattern, so it
    /// must be the same whichever way up the marker is photographed.
    #[test]
    fn dictionary_free_ids_are_rotation_invariant() {
        let img = render_marker(7, 20, 40, 40, 300);
        let params = ArucoParams::default();
        assert!(params.dictionary_free, "dictionary-free is the default");

        let id_of = |im: &image::DynamicImage| -> Option<u32> {
            let fs = detect(im, &params);
            (fs.len() == 4).then(|| fs.descriptors.marker_corner(0).unwrap().1)
        };
        let upright = id_of(&img).expect("marker detected upright");
        for turns in 1..4 {
            let mut r = img.clone();
            for _ in 0..turns {
                r = image::DynamicImage::ImageLuma8(image::imageops::rotate90(&r.to_luma8()));
            }
            assert_eq!(
                id_of(&r),
                Some(upright),
                "id changed after {turns} quarter turn(s)"
            );
        }
    }

    #[test]
    fn rotating_a_word_four_times_is_the_identity() {
        for n in [4usize, 5, 6] {
            let mask = if n * n == 64 {
                u64::MAX
            } else {
                (1u64 << (n * n)) - 1
            };
            for seed in [0x9E37u64, 0x1234, 0xFFFF, 1, 0] {
                let w = seed & mask;
                let mut r = w;
                for _ in 0..4 {
                    r = rotate_word(r, n);
                }
                assert_eq!(r, w, "n={n} seed={seed:#x}");
            }
        }
    }

    #[test]
    fn canonical_form_is_the_same_for_every_rotation() {
        let n = 4;
        let word = 0b0110_1001_0011_1100u64;
        let (want, _) = canonical_word(word, n);
        let mut w = word;
        for turn in 0..4 {
            let (got, turns) = canonical_word(w, n);
            assert_eq!(got, want, "rotation {turn} disagreed");
            // Applying the reported turns must actually reach the canonical form.
            let mut c = w;
            for _ in 0..turns {
                c = rotate_word(c, n);
            }
            assert_eq!(c, want);
            w = rotate_word(w, n);
        }
    }

    /// A 5x5 board must be read as 5x5 without being told.
    #[test]
    fn the_data_grid_size_is_detected_rather_than_assumed() {
        // A 5x5 marker: black ring plus a 5x5 interior. The data cells next to
        // the border are paper, so reading this on a 4x4 grid pulls white into
        // what would have to be a solid ink ring and fails, which is the
        // discrimination being tested.
        // Deliberately not a closed ring: a ring of ink inside the border is
        // itself a valid square fiducial, so such a pattern produces a nested
        // second candidate and makes the test about the wrong thing.
        let bits = [
            [false, false, false, false, false],
            [false, true, true, false, false],
            [false, false, true, false, false],
            [false, false, true, true, false],
            [false, false, false, false, false],
        ];
        let grid = 7;
        let cell = 24u32;
        let margin = 40u32;
        let size = margin * 2 + cell * grid as u32;
        let mut img = image::GrayImage::from_pixel(size, size, image::Luma([255]));
        for r in 0..grid {
            for c in 0..grid {
                let ink = if r == 0 || c == 0 || r == grid - 1 || c == grid - 1 {
                    true
                } else {
                    bits[r - 1][c - 1]
                };
                if !ink {
                    continue;
                }
                for y in 0..cell {
                    for x in 0..cell {
                        img.put_pixel(
                            margin + c as u32 * cell + x,
                            margin + r as u32 * cell + y,
                            image::Luma([0]),
                        );
                    }
                }
            }
        }
        let dynimg = image::DynamicImage::ImageLuma8(img);
        assert_eq!(
            detect_family(&dynimg, &ArucoParams::default()),
            Some(5),
            "a 5x5 board should be recognised as 5x5"
        );
        assert_eq!(detect(&dynimg, &ArucoParams::default()).len(), 4);
    }

    /// The defect that produced confident, wrong ids: a bound past what the
    /// code can correct.
    #[test]
    fn the_hamming_bound_cannot_exceed_the_codes_correction_capability() {
        assert_eq!(MAX_CORRECTABLE_ERRORS, (DICT_MIN_DISTANCE - 1) / 2);
        assert_eq!(
            ArucoParams::default().max_hamming_distance,
            MAX_CORRECTABLE_ERRORS
        );
        // Asking for more must not widen what gets accepted.
        let img = render_marker(3, 20, 40, 40, 300);
        let greedy = ArucoParams {
            dictionary_free: false,
            data_bits: Some(DATA_BITS),
            max_hamming_distance: 9,
            ..Default::default()
        };
        assert_eq!(detect(&img, &greedy).len(), 4);
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
