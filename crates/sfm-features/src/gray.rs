//! Shared grayscale image utilities used by all detectors.

use image::{DynamicImage, ImageBuffer, Luma};

pub type GrayF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

/// Convert any image to a single-channel `f32` image normalized to `[0, 1]`.
pub fn to_gray_f32(img: &DynamicImage) -> GrayF32 {
    let luma = img.to_luma8();
    ImageBuffer::from_fn(luma.width(), luma.height(), |x, y| {
        Luma([luma.get_pixel(x, y).0[0] as f32 / 255.0])
    })
}

/// 2x2 box-average downsample. Used to build the SIFT/ORB image pyramids.
pub fn downsample2x(img: &GrayF32) -> GrayF32 {
    let (w, h) = img.dimensions();
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    ImageBuffer::from_fn(nw, nh, |x, y| {
        let x0 = (x * 2).min(w - 1);
        let y0 = (y * 2).min(h - 1);
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let sum = img.get_pixel(x0, y0).0[0]
            + img.get_pixel(x1, y0).0[0]
            + img.get_pixel(x0, y1).0[0]
            + img.get_pixel(x1, y1).0[0];
        Luma([sum * 0.25])
    })
}

/// Bilinear 2x upsample. Lowe's original SIFT construction doubles the input
/// before building the first octave, trading compute for the ability to
/// detect low-contrast/small keypoints that don't survive even one round of
/// Gaussian blur + downsampling at native resolution - the effect is most
/// visible on already-small images, where skipping it leaves very little
/// resolution for anything past the first octave or two.
pub fn upsample2x(img: &GrayF32) -> GrayF32 {
    let (w, h) = img.dimensions();
    let nw = w * 2;
    let nh = h * 2;
    ImageBuffer::from_fn(nw, nh, |x, y| {
        Luma([sample_bilinear(img, x as f32 / 2.0, y as f32 / 2.0)])
    })
}

pub fn gaussian_blur(img: &GrayF32, sigma: f32) -> GrayF32 {
    imageproc::filter::gaussian_blur_f32(img, sigma)
}

/// Bilinear sample, clamping to image bounds (returns `0.0` only for a
/// zero-sized image, which never occurs on real input).
pub fn sample_bilinear(img: &GrayF32, x: f32, y: f32) -> f32 {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let v00 = img.get_pixel(x0, y0).0[0];
    let v10 = img.get_pixel(x1, y0).0[0];
    let v01 = img.get_pixel(x0, y1).0[0];
    let v11 = img.get_pixel(x1, y1).0[0];
    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;
    v0 * (1.0 - fy) + v1 * fy
}
