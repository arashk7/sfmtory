//! Native ORB (Rublee et al., ICCV 2011): multi-scale FAST-9 corners ranked
//! by a simple corner-strength score, intensity-centroid orientation, and a
//! rotated (steered) BRIEF-256 binary descriptor. Original implementation;
//! the BRIEF sampling pattern is our own deterministically-seeded pattern
//! (uniform in a disk) rather than ORB's greedily-learned pattern - close
//! enough in practice, and avoids depending on any pre-trained/pre-licensed
//! pattern table. Free of the SIFT/SURF patent concerns entirely (ORB was
//! designed as a patent-free alternative).

use std::sync::OnceLock;

use sfm_core::{Descriptors, FeatureSet, Keypoint};

use crate::gray::{sample_bilinear, to_gray_f32, GrayF32};

#[derive(Debug, Clone, Copy)]
pub struct OrbParams {
    pub num_levels: u32,
    pub scale_factor: f32,
    /// Intensity difference required for the FAST circle test (0..255 scale).
    pub fast_threshold: u8,
    pub patch_radius: i32,
    pub max_features: Option<usize>,
}

impl Default for OrbParams {
    fn default() -> Self {
        OrbParams {
            num_levels: 8,
            scale_factor: 1.2,
            fast_threshold: 20,
            patch_radius: 15,
            max_features: Some(8000),
        }
    }
}

const CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

pub fn detect(img: &image::DynamicImage, params: &OrbParams) -> FeatureSet {
    let base = to_gray_f32(img);
    let pattern = brief_pattern();

    let mut keypoints = Vec::new();
    let mut desc_bytes: Vec<u8> = Vec::new();

    let mut level_img = base;
    let mut level_scale = 1.0f32;
    for _level in 0..params.num_levels {
        let (w, h) = level_img.dimensions();
        let border = params.patch_radius + 3;
        if (w as i32) < 2 * border + 1 || (h as i32) < 2 * border + 1 {
            break;
        }

        let scores = fast_score_map(&level_img, params.fast_threshold);
        let corners = non_max_suppress(&scores, w, h, border);

        for (x, y, _score) in &corners {
            let angle = match intensity_centroid_angle(&level_img, *x, *y, params.patch_radius) {
                Some(a) => a,
                None => continue,
            };
            let descriptor = compute_brief(&level_img, *x, *y, angle, &pattern);

            keypoints.push(Keypoint {
                x: *x as f32 * level_scale,
                y: *y as f32 * level_scale,
                scale: level_scale,
                angle,
                response: scores[(*y as usize) * w as usize + *x as usize],
            });
            desc_bytes.extend_from_slice(&descriptor);
        }

        level_scale *= params.scale_factor;
        let nw = (w as f32 / params.scale_factor).round().max(1.0) as u32;
        let nh = (h as f32 / params.scale_factor).round().max(1.0) as u32;
        if nw < 2 * border as u32 + 1 || nh < 2 * border as u32 + 1 {
            break;
        }
        level_img = resize_bilinear(&level_img, nw, nh);
    }

    let mut feature_set = FeatureSet {
        keypoints,
        descriptors: Descriptors::Binary {
            bytes_per_descriptor: 32,
            data: desc_bytes,
        },
    };
    if let Some(max) = params.max_features {
        feature_set.truncate_to_strongest(max);
    }
    feature_set
}

fn get_u8(img: &GrayF32, x: i32, y: i32) -> f32 {
    img.get_pixel(x as u32, y as u32).0[0] * 255.0
}

fn fast_score_map(img: &GrayF32, threshold: u8) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let mut scores = vec![0f32; (w * h) as usize];
    let border = 3;
    let t = threshold as f32;
    for y in border..(h as i32 - border) {
        for x in border..(w as i32 - border) {
            let center = get_u8(img, x, y);
            let mut states = [0i8; 16];
            for (i, (dx, dy)) in CIRCLE.iter().enumerate() {
                let v = get_u8(img, x + dx, y + dy);
                states[i] = if v >= center + t {
                    1
                } else if v <= center - t {
                    -1
                } else {
                    0
                };
            }
            if let Some(score) = corner_score(&states, img, x, y, center, t) {
                scores[(y as usize) * w as usize + x as usize] = score;
            }
        }
    }
    scores
}

fn corner_score(
    states: &[i8; 16],
    img: &GrayF32,
    x: i32,
    y: i32,
    center: f32,
    t: f32,
) -> Option<f32> {
    let doubled: Vec<i8> = states.iter().chain(states.iter()).copied().collect();
    let (mut best_bright, mut cur_bright) = (0usize, 0usize);
    let (mut best_dark, mut cur_dark) = (0usize, 0usize);
    for &s in &doubled {
        if s == 1 {
            cur_bright += 1;
            best_bright = best_bright.max(cur_bright);
        } else {
            cur_bright = 0;
        }
        if s == -1 {
            cur_dark += 1;
            best_dark = best_dark.max(cur_dark);
        } else {
            cur_dark = 0;
        }
    }
    if best_bright < 9 && best_dark < 9 {
        return None;
    }
    let mut score = 0f32;
    for (dx, dy) in CIRCLE.iter() {
        let v = get_u8(img, x + dx, y + dy);
        let d = (v - center).abs() - t;
        if d > 0.0 {
            score += d;
        }
    }
    Some(score)
}

fn non_max_suppress(scores: &[f32], w: u32, h: u32, border: i32) -> Vec<(i32, i32, f32)> {
    let mut out = Vec::new();
    for y in border..(h as i32 - border) {
        for x in border..(w as i32 - border) {
            let s = scores[(y as usize) * w as usize + x as usize];
            if s <= 0.0 {
                continue;
            }
            let mut is_max = true;
            'outer: for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    if scores[(ny as usize) * w as usize + nx as usize] > s {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                out.push((x, y, s));
            }
        }
    }
    out
}

fn intensity_centroid_angle(img: &GrayF32, x: i32, y: i32, radius: i32) -> Option<f32> {
    let (w, h) = img.dimensions();
    if x - radius < 0 || y - radius < 0 || x + radius >= w as i32 || y + radius >= h as i32 {
        return None;
    }
    let mut m10 = 0f32;
    let mut m01 = 0f32;
    let r2 = (radius * radius) as f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if (dx * dx + dy * dy) as f32 > r2 {
                continue;
            }
            let v = get_u8(img, x + dx, y + dy);
            m10 += dx as f32 * v;
            m01 += dy as f32 * v;
        }
    }
    Some(m01.atan2(m10))
}

fn compute_brief(
    img: &GrayF32,
    x: i32,
    y: i32,
    angle: f32,
    pattern: &[(f32, f32, f32, f32); 256],
) -> [u8; 32] {
    let cos_t = angle.cos();
    let sin_t = angle.sin();
    let mut bytes = [0u8; 32];
    for (i, (x1, y1, x2, y2)) in pattern.iter().enumerate() {
        let rx1 = x1 * cos_t - y1 * sin_t;
        let ry1 = x1 * sin_t + y1 * cos_t;
        let rx2 = x2 * cos_t - y2 * sin_t;
        let ry2 = x2 * sin_t + y2 * cos_t;
        let v1 = sample_bilinear(img, x as f32 + rx1, y as f32 + ry1);
        let v2 = sample_bilinear(img, x as f32 + rx2, y as f32 + ry2);
        if v1 < v2 {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

/// Deterministic xorshift-generated BRIEF sampling pattern: 256 pairs of
/// `(x1, y1, x2, y2)` offsets, uniform within a disk of radius `patch_radius`.
/// Fixed seed so the pattern (and therefore descriptor bit meaning) is stable
/// across runs/builds.
fn brief_pattern() -> &'static [(f32, f32, f32, f32); 256] {
    static PATTERN: OnceLock<[(f32, f32, f32, f32); 256]> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut rand_offset = || loop {
            let bits = next();
            let fx = ((bits & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
            let fy = (((bits >> 16) & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
            if fx * fx + fy * fy <= 1.0 {
                return (fx * 15.0, fy * 15.0);
            }
        };
        let mut pattern = [(0f32, 0f32, 0f32, 0f32); 256];
        for p in pattern.iter_mut() {
            let (x1, y1) = rand_offset();
            let (x2, y2) = rand_offset();
            *p = (x1, y1, x2, y2);
        }
        pattern
    })
}

fn resize_bilinear(img: &GrayF32, new_w: u32, new_h: u32) -> GrayF32 {
    let (w, h) = img.dimensions();
    let sx = w as f32 / new_w as f32;
    let sy = h as f32 / new_h as f32;
    image::ImageBuffer::from_fn(new_w, new_h, |x, y| {
        let src_x = (x as f32 + 0.5) * sx - 0.5;
        let src_y = (y as f32 + 0.5) * sy - 0.5;
        image::Luma([sample_bilinear(img, src_x, src_y)])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn synthetic_scene(w: u32, h: u32) -> DynamicImage {
        let img = GrayImage::from_fn(w, h, |x, y| {
            let checker = (((x / 10) + (y / 10)) % 2) as f32 * 255.0;
            Luma([checker as u8])
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn finds_corners_on_checkerboard() {
        let img = synthetic_scene(200, 200);
        let features = detect(&img, &OrbParams::default());
        assert!(
            !features.is_empty(),
            "expected ORB to find corners on a checkerboard"
        );
        assert_eq!(features.descriptors.len(), features.keypoints.len());
        if let Descriptors::Binary {
            bytes_per_descriptor,
            ..
        } = &features.descriptors
        {
            assert_eq!(*bytes_per_descriptor, 32);
        } else {
            panic!("expected binary descriptors");
        }
    }

    #[test]
    fn finds_no_corners_on_blank_image() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(128, 128, Luma([128])));
        let features = detect(&img, &OrbParams::default());
        assert!(features.is_empty());
    }
}
