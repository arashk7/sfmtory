//! Native SIFT (Lowe, IJCV 2004): Gaussian/DoG scale-space extrema, subpixel
//! + edge-response filtering, orientation assignment, and the 128-d
//! trilinear-interpolated gradient-histogram descriptor. SIFT's patent
//! (US6711293) expired in 2020, so this is free to use commercially; this is
//! an original implementation (no code taken from any existing SIFT).
//!
//! One deliberate deviation from Lowe's original: no 2x pre-upsampling of the
//! input image (which he uses to find more low-contrast keypoints at the cost
//! of ~4x the compute). Trading a little recall on tiny/blurry features for
//! speed matches this project's "as fast as possible" goal; upsampling can be
//! added later as a `--detector sift --upsample` flag if a scene needs it.

use nalgebra::{Matrix3, Vector3};
use sfm_core::{Descriptors, FeatureSet, Keypoint};

use crate::gray::{downsample2x, gaussian_blur, to_gray_f32, upsample2x, GrayF32};

#[derive(Debug, Clone, Copy)]
pub struct SiftParams {
    pub scales_per_octave: u32,
    pub sigma0: f32,
    pub contrast_threshold: f32,
    pub edge_threshold: f32,
    pub max_features: Option<usize>,
    /// Double the image before building the first octave (Lowe's original
    /// construction, also COLMAP's/VLFeat's default) - finds substantially
    /// more low-contrast/small keypoints, at roughly 4x the compute *and
    /// memory* of the first octave. Matters most on already-small images:
    /// skipping it leaves very little resolution for anything past the
    /// first octave or two, which measurably starved feature counts on
    /// `temple_sparse_ring` (640x480 photos) specifically - see PLAN.md.
    /// Only actually applied below `UPSAMPLE_MAX_MIN_DIM` (see `detect`):
    /// full-resolution photos (`sceaux_castle`'s ~2832x2128 originals) have
    /// no shortage of resolution to begin with, and the same 4x blowup that
    /// helps a small image OOM-crashed this exact environment when applied
    /// unconditionally to 11 of them extracted in parallel.
    pub upsample: bool,
}

/// Upsampling is skipped above this original min-dimension regardless of
/// `SiftParams::upsample` - see that field's docs.
const UPSAMPLE_MAX_MIN_DIM: u32 = 1600;

impl Default for SiftParams {
    fn default() -> Self {
        SiftParams {
            scales_per_octave: 3,
            sigma0: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            max_features: Some(8000),
            upsample: true,
        }
    }
}

struct RawKeypoint {
    octave: u32,
    /// Position and scale index within the octave's own (downsampled) pixel grid.
    x: f32,
    y: f32,
    s: f32,
    response: f32,
}

pub fn detect(img: &image::DynamicImage, params: &SiftParams) -> FeatureSet {
    let base = to_gray_f32(img);
    let (w0, h0) = base.dimensions();
    let min_dim = w0.min(h0);
    if min_dim < 16 {
        return FeatureSet {
            keypoints: Vec::new(),
            descriptors: Descriptors::Float32 {
                dim: 128,
                data: Vec::new(),
            },
        };
    }

    // Upsampling doubles the sampling rate, so the same physical input blur
    // corresponds to double the assumed sigma in the new (upsampled) pixel
    // grid - and every final keypoint coordinate/scale needs converting back
    // to original-image pixels by `coord_scale` at the end.
    let should_upsample = params.upsample && min_dim <= UPSAMPLE_MAX_MIN_DIM;
    let (base, assumed_input_sigma, coord_scale) = if should_upsample {
        (upsample2x(&base), 1.0f32, 0.5f32)
    } else {
        (base, 0.5f32, 1.0f32)
    };
    let sigma_diff = (params.sigma0 * params.sigma0 - assumed_input_sigma * assumed_input_sigma)
        .max(0.01)
        .sqrt();
    let base = gaussian_blur(&base, sigma_diff);

    let (working_w, working_h) = base.dimensions();
    let working_min_dim = working_w.min(working_h);
    let num_octaves = (((working_min_dim as f32).log2().floor() as i32) - 2).clamp(1, 8) as usize;
    let s = params.scales_per_octave;
    let k = 2f32.powf(1.0 / s as f32);

    // gaussian[octave][scale_index], scale_index in 0..s+3.
    let mut gaussian: Vec<Vec<GrayF32>> = Vec::with_capacity(num_octaves);
    // Per-scale total sigma relative to that octave's base image resolution.
    let sigma_total: Vec<f32> = (0..=s + 2)
        .map(|i| params.sigma0 * k.powi(i as i32))
        .collect();

    let mut current = base;
    for _oct in 0..num_octaves {
        let mut octave_images = Vec::with_capacity((s + 3) as usize);
        octave_images.push(current.clone());
        for i in 1..=(s + 2) {
            let prev_sigma = sigma_total[(i - 1) as usize];
            let this_sigma = sigma_total[i as usize];
            let incremental = (this_sigma * this_sigma - prev_sigma * prev_sigma)
                .max(0.01)
                .sqrt();
            let blurred = gaussian_blur(&octave_images[(i - 1) as usize], incremental);
            octave_images.push(blurred);
        }
        // Seed the next octave from the image at sigma = 2*sigma0 (index `s`),
        // matching Lowe's construction so scale sampling stays continuous
        // across the octave boundary.
        current = downsample2x(&octave_images[s as usize]);
        gaussian.push(octave_images);
    }

    let dog: Vec<Vec<GrayF32>> = gaussian
        .iter()
        .map(|oct| {
            (0..oct.len() - 1)
                .map(|i| diff_image(&oct[i + 1], &oct[i]))
                .collect()
        })
        .collect();

    let mut raw_keypoints = Vec::new();
    for (o, oct_dog) in dog.iter().enumerate() {
        find_octave_extrema(oct_dog, o as u32, s, params, &mut raw_keypoints);
    }

    let mut keypoints = Vec::new();
    let mut desc_rows: Vec<[f32; 128]> = Vec::new();
    for rk in &raw_keypoints {
        let oct_gaussian = &gaussian[rk.octave as usize];
        let scale_img_idx = rk.s.round().clamp(0.0, (oct_gaussian.len() - 1) as f32) as usize;
        let img = &oct_gaussian[scale_img_idx];
        let sigma_octave = params.sigma0 * k.powf(rk.s);

        let angle = match dominant_orientation(img, rk.x, rk.y, sigma_octave) {
            Some(a) => a,
            None => continue,
        };
        let descriptor = match compute_descriptor(img, rk.x, rk.y, sigma_octave, angle) {
            Some(d) => d,
            None => continue,
        };

        let scale_factor = 2f32.powi(rk.octave as i32) * coord_scale;
        keypoints.push(Keypoint {
            x: rk.x * scale_factor,
            y: rk.y * scale_factor,
            scale: sigma_octave * scale_factor,
            angle,
            response: rk.response,
        });
        desc_rows.push(descriptor);
    }

    let mut feature_set = FeatureSet {
        keypoints,
        descriptors: Descriptors::Float32 {
            dim: 128,
            data: desc_rows.into_iter().flatten().collect(),
        },
    };
    if let Some(max) = params.max_features {
        feature_set.truncate_to_strongest(max);
    }
    feature_set
}

fn diff_image(a: &GrayF32, b: &GrayF32) -> GrayF32 {
    let (w, h) = a.dimensions();
    image::ImageBuffer::from_fn(w, h, |x, y| {
        image::Luma([a.get_pixel(x, y).0[0] - b.get_pixel(x, y).0[0]])
    })
}

const BORDER: i32 = 5;
const MAX_REFINE_ITERS: usize = 5;

fn find_octave_extrema(
    oct_dog: &[GrayF32],
    octave: u32,
    scales_per_octave: u32,
    params: &SiftParams,
    out: &mut Vec<RawKeypoint>,
) {
    let (w, h) = oct_dog[0].dimensions();
    if w as i32 <= 2 * BORDER || h as i32 <= 2 * BORDER {
        return;
    }
    let prelim_threshold = 0.5 * params.contrast_threshold / scales_per_octave as f32;

    for si in 1..oct_dog.len() - 1 {
        for y in BORDER..(h as i32 - BORDER) {
            for x in BORDER..(w as i32 - BORDER) {
                let center = pix(&oct_dog[si], x, y);
                if center.abs() <= prelim_threshold {
                    continue;
                }
                if !is_local_extremum(oct_dog, si, x, y, center) {
                    continue;
                }
                if let Some((rx, ry, rs, contrast)) =
                    refine_extremum(oct_dog, si, x, y, params, scales_per_octave)
                {
                    if passes_edge_test(
                        &oct_dog[si],
                        rx.round() as i32,
                        ry.round() as i32,
                        params.edge_threshold,
                    ) {
                        out.push(RawKeypoint {
                            octave,
                            x: rx,
                            y: ry,
                            s: rs,
                            response: contrast.abs(),
                        });
                    }
                }
            }
        }
    }
}

fn pix(img: &GrayF32, x: i32, y: i32) -> f32 {
    img.get_pixel(x as u32, y as u32).0[0]
}

fn is_local_extremum(oct_dog: &[GrayF32], si: usize, x: i32, y: i32, center: f32) -> bool {
    let is_max = center > 0.0;
    for ds in -1..=1i32 {
        let img = &oct_dog[(si as i32 + ds) as usize];
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if ds == 0 && dx == 0 && dy == 0 {
                    continue;
                }
                let v = pix(img, x + dx, y + dy);
                if is_max {
                    if v >= center {
                        return false;
                    }
                } else if v <= center {
                    return false;
                }
            }
        }
    }
    true
}

/// Iterative Taylor-expansion subpixel localization (Brown & Lowe 2002).
/// Returns `(x, y, scale_index, contrast)` in the octave's own pixel grid.
fn refine_extremum(
    oct_dog: &[GrayF32],
    mut si: usize,
    mut x: i32,
    mut y: i32,
    params: &SiftParams,
    scales_per_octave: u32,
) -> Option<(f32, f32, f32, f32)> {
    let (w, h) = oct_dog[0].dimensions();
    for _ in 0..MAX_REFINE_ITERS {
        let dx = (pix(&oct_dog[si], x + 1, y) - pix(&oct_dog[si], x - 1, y)) * 0.5;
        let dy = (pix(&oct_dog[si], x, y + 1) - pix(&oct_dog[si], x, y - 1)) * 0.5;
        let ds = (pix(&oct_dog[si + 1], x, y) - pix(&oct_dog[si - 1], x, y)) * 0.5;
        let grad = Vector3::new(dx as f64, dy as f64, ds as f64);

        let center = pix(&oct_dog[si], x, y);
        let dxx = pix(&oct_dog[si], x + 1, y) + pix(&oct_dog[si], x - 1, y) - 2.0 * center;
        let dyy = pix(&oct_dog[si], x, y + 1) + pix(&oct_dog[si], x, y - 1) - 2.0 * center;
        let dss = pix(&oct_dog[si + 1], x, y) + pix(&oct_dog[si - 1], x, y) - 2.0 * center;
        let dxy = (pix(&oct_dog[si], x + 1, y + 1)
            - pix(&oct_dog[si], x + 1, y - 1)
            - pix(&oct_dog[si], x - 1, y + 1)
            + pix(&oct_dog[si], x - 1, y - 1))
            * 0.25;
        let dxs = (pix(&oct_dog[si + 1], x + 1, y)
            - pix(&oct_dog[si + 1], x - 1, y)
            - pix(&oct_dog[si - 1], x + 1, y)
            + pix(&oct_dog[si - 1], x - 1, y))
            * 0.25;
        let dys = (pix(&oct_dog[si + 1], x, y + 1)
            - pix(&oct_dog[si + 1], x, y - 1)
            - pix(&oct_dog[si - 1], x, y + 1)
            + pix(&oct_dog[si - 1], x, y - 1))
            * 0.25;
        let hessian = Matrix3::new(
            dxx as f64, dxy as f64, dxs as f64, dxy as f64, dyy as f64, dys as f64, dxs as f64,
            dys as f64, dss as f64,
        );

        let offset = match hessian.try_inverse() {
            Some(inv) => -(inv * grad),
            None => return None,
        };

        if offset.x.abs() < 0.6 && offset.y.abs() < 0.6 && offset.z.abs() < 0.6 {
            let contrast = center as f64 + 0.5 * grad.dot(&offset);
            let effective_threshold = params.contrast_threshold as f64 / scales_per_octave as f64;
            if contrast.abs() < effective_threshold {
                return None;
            }
            let fx = x as f32 + offset.x as f32;
            let fy = y as f32 + offset.y as f32;
            let fs = si as f32 + offset.z as f32;
            if fx < BORDER as f32
                || fy < BORDER as f32
                || fx > (w as i32 - BORDER - 1) as f32
                || fy > (h as i32 - BORDER - 1) as f32
            {
                return None;
            }
            return Some((fx, fy, fs, contrast as f32));
        }

        x += offset.x.round() as i32;
        y += offset.y.round() as i32;
        let new_si =
            (si as i32 + offset.z.round() as i32).clamp(1, oct_dog.len() as i32 - 2) as usize;
        si = new_si;
        if x <= BORDER || y <= BORDER || x >= w as i32 - BORDER - 1 || y >= h as i32 - BORDER - 1 {
            return None;
        }
    }
    None
}

fn passes_edge_test(img: &GrayF32, x: i32, y: i32, edge_threshold: f32) -> bool {
    let center = pix(img, x, y);
    let dxx = pix(img, x + 1, y) + pix(img, x - 1, y) - 2.0 * center;
    let dyy = pix(img, x, y + 1) + pix(img, x, y - 1) - 2.0 * center;
    let dxy = (pix(img, x + 1, y + 1) - pix(img, x + 1, y - 1) - pix(img, x - 1, y + 1)
        + pix(img, x - 1, y - 1))
        * 0.25;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 {
        return false;
    }
    let r = edge_threshold;
    (tr * tr) / det < (r + 1.0) * (r + 1.0) / r
}

fn gradient_at(img: &GrayF32, x: i32, y: i32) -> Option<(f32, f32)> {
    let (w, h) = img.dimensions();
    if x < 1 || y < 1 || x >= w as i32 - 1 || y >= h as i32 - 1 {
        return None;
    }
    let dx = pix(img, x + 1, y) - pix(img, x - 1, y);
    let dy = pix(img, x, y + 1) - pix(img, x, y - 1);
    let mag = (dx * dx + dy * dy).sqrt();
    let ori = dy.atan2(dx);
    Some((mag, ori))
}

fn dominant_orientation(img: &GrayF32, x: f32, y: f32, sigma_octave: f32) -> Option<f32> {
    const NBINS: usize = 36;
    let radius = (3.0 * 1.5 * sigma_octave).round() as i32;
    if radius <= 0 {
        return None;
    }
    let mut hist = [0f32; NBINS];
    let gauss_sigma = 1.5 * sigma_octave;
    let xi = x.round() as i32;
    let yi = y.round() as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = xi + dx;
            let py = yi + dy;
            if let Some((mag, ori)) = gradient_at(img, px, py) {
                let weight =
                    (-((dx * dx + dy * dy) as f32) / (2.0 * gauss_sigma * gauss_sigma)).exp();
                let mut bin = ((ori.to_degrees() + 360.0) % 360.0) / (360.0 / NBINS as f32);
                if bin >= NBINS as f32 {
                    bin -= NBINS as f32;
                }
                hist[bin as usize] += weight * mag;
            }
        }
    }
    // Smooth the circular histogram to reduce quantization noise.
    for _ in 0..2 {
        let mut smoothed = [0f32; NBINS];
        for i in 0..NBINS {
            let prev = hist[(i + NBINS - 1) % NBINS];
            let next = hist[(i + 1) % NBINS];
            smoothed[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * next;
        }
        hist = smoothed;
    }
    let (max_idx, &max_val) = hist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
    if max_val <= 0.0 {
        return None;
    }
    let bin_width = 360.0 / NBINS as f32;
    Some((max_idx as f32 * bin_width).to_radians())
}

fn compute_descriptor(
    img: &GrayF32,
    x: f32,
    y: f32,
    sigma_octave: f32,
    angle: f32,
) -> Option<[f32; 128]> {
    const NBINS_ORI: usize = 8;
    const NBP: usize = 4; // 4x4 spatial bins
    let cos_t = angle.cos();
    let sin_t = angle.sin();
    let hist_width = 3.0 * sigma_octave;
    let radius = (hist_width * std::f32::consts::SQRT_2 * (NBP as f32 + 1.0) * 0.5).round() as i32;
    if radius <= 0 {
        return None;
    }

    let mut hist = [[[0f32; NBINS_ORI]; NBP]; NBP];
    let bins_per_rad = NBINS_ORI as f32 / (2.0 * std::f32::consts::PI);
    let exp_denom = 2.0 * (NBP as f32 * 0.5) * (NBP as f32 * 0.5);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            // Rotate into the keypoint's own frame, normalized by bin width.
            let rot_x = (dx as f32 * cos_t + dy as f32 * sin_t) / hist_width;
            let rot_y = (-(dx as f32) * sin_t + dy as f32 * cos_t) / hist_width;
            let rbin = rot_y + NBP as f32 / 2.0 - 0.5;
            let cbin = rot_x + NBP as f32 / 2.0 - 0.5;
            if rbin <= -1.0 || rbin >= NBP as f32 || cbin <= -1.0 || cbin >= NBP as f32 {
                continue;
            }
            let px = x.round() as i32 + dx;
            let py = y.round() as i32 + dy;
            let (mag, ori) = match gradient_at(img, px, py) {
                Some(v) => v,
                None => continue,
            };
            let rel_ori = {
                let mut o = ori - angle;
                while o < 0.0 {
                    o += 2.0 * std::f32::consts::PI;
                }
                while o >= 2.0 * std::f32::consts::PI {
                    o -= 2.0 * std::f32::consts::PI;
                }
                o
            };
            let obin = rel_ori * bins_per_rad;
            let weight = (-(rot_x * rot_x + rot_y * rot_y) / exp_denom).exp() * mag;

            trilinear_add(&mut hist, rbin, cbin, obin, weight);
        }
    }

    let mut flat = [0f32; 128];
    let mut idx = 0;
    for r in 0..NBP {
        for c in 0..NBP {
            for o in 0..NBINS_ORI {
                flat[idx] = hist[r][c][o];
                idx += 1;
            }
        }
    }
    let norm: f32 = flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-6 {
        return None;
    }
    for v in flat.iter_mut() {
        *v = (*v / norm).min(0.2);
    }
    let norm2: f32 = flat.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    for v in flat.iter_mut() {
        *v /= norm2;
    }
    Some(flat)
}

fn trilinear_add(hist: &mut [[[f32; 8]; 4]; 4], rbin: f32, cbin: f32, obin: f32, weight: f32) {
    let r0 = rbin.floor();
    let c0 = cbin.floor();
    let o0 = obin.floor();
    let rf = rbin - r0;
    let cf = cbin - c0;
    let of = obin - o0;

    for (dr, wr) in [(0, 1.0 - rf), (1, rf)] {
        let r = r0 as i32 + dr;
        if r < 0 || r >= 4 {
            continue;
        }
        for (dc, wc) in [(0, 1.0 - cf), (1, cf)] {
            let c = c0 as i32 + dc;
            if c < 0 || c >= 4 {
                continue;
            }
            for (do_, wo) in [(0, 1.0 - of), (1, of)] {
                let o = ((o0 as i32 + do_).rem_euclid(8)) as usize;
                hist[r as usize][c as usize][o] += weight * wr * wc * wo;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    /// A textured synthetic scene (checkerboard-ish blobs) so DoG extrema
    /// actually exist - a blank or single-gradient image has none.
    fn synthetic_scene(w: u32, h: u32) -> DynamicImage {
        let img = GrayImage::from_fn(w, h, |x, y| {
            let v = (((x / 12) + (y / 12)) % 2) as f32 * 255.0;
            let blob = (((x as f32 - w as f32 / 2.0).powi(2)
                + (y as f32 - h as f32 / 2.0).powi(2))
                / 400.0)
                .exp()
                .recip()
                * 255.0;
            Luma([((v * 0.6 + blob * 0.4).clamp(0.0, 255.0)) as u8])
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn finds_keypoints_on_textured_image() {
        let img = synthetic_scene(256, 256);
        let params = SiftParams::default();
        let features = detect(&img, &params);
        assert!(
            !features.is_empty(),
            "expected SIFT to find keypoints on a textured image"
        );
        assert_eq!(features.descriptors.len(), features.keypoints.len());
        if let Descriptors::Float32 { dim, .. } = &features.descriptors {
            assert_eq!(*dim, 128);
        } else {
            panic!("expected float descriptors");
        }
        // Every descriptor should be unit-normalized.
        for i in 0..features.len() {
            let row = features.descriptors.float_row(i).unwrap();
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "descriptor {i} not unit-normalized: {norm}"
            );
        }
    }

    #[test]
    fn finds_no_keypoints_on_blank_image() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(128, 128, Luma([128])));
        let features = detect(&img, &SiftParams::default());
        assert!(features.is_empty());
    }
}
