//! Shared grayscale image utilities used by all detectors.

use image::{DynamicImage, GrayImage, ImageBuffer, Luma};

pub type GrayF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

/// Rec.709 luma, parallelised over rows.
///
/// The conversion is trivial per pixel, but at 12 megapixels the serial version
/// was the single largest stage of ArUco detection once the real algorithmic
/// problem was fixed.
///
/// This is deliberately *our* definition of luma rather than a reproduction of
/// `DynamicImage::to_luma8`. Matching that bit-for-bit was tried and abandoned:
/// it routes through a private colour-space cast with float weights, so any
/// reimplementation is a guess pinned to one release of `image` that a version
/// bump could silently invalidate - and silently changing the grey levels the
/// adaptive threshold works in is exactly the kind of breakage that shows up as
/// "detection got worse on some images" months later.
///
/// The integer form here (Rec.709 weights, rounded half-up) reproduces greys
/// exactly and differs from `to_luma8` by at most one level on colours. That it does not change detection is verified
/// against a real 12MP frame rather than assumed - see the aruco tests. Note
/// `Pixel::to_luma` is a third, different function: it truncates, and disagreed
/// on 45% of pixels.
pub fn to_luma8_par(img: &DynamicImage) -> GrayImage {
    use rayon::prelude::*;

    /// Rec.709 luma, x10000, as the `image` crate stores them.
    const LUMA: [u32; 3] = [2126, 7152, 722];

    let Some(rgb) = img.as_rgb8() else {
        // Anything not already 8-bit RGB (greyscale input, 16-bit, RGBA) is
        // rare enough on camera input not to warrant a second path.
        return img.to_luma8();
    };
    let (w, h) = rgb.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let src = rgb.as_raw();
    let mut out = vec![0u8; wu * hu];
    out.par_chunks_mut(wu).enumerate().for_each(|(y, row)| {
        let base = y * wu * 3;
        for (x, o) in row.iter_mut().enumerate() {
            let p = &src[base + x * 3..base + x * 3 + 3];
            let sum = LUMA[0] * p[0] as u32 + LUMA[1] * p[1] as u32 + LUMA[2] * p[2] as u32;
            *o = ((sum + 5000) / 10000) as u8;
        }
    });
    GrayImage::from_raw(w, h, out).expect("buffer sized from the source dimensions")
}

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

#[cfg(test)]
mod par_luma_tests {
    use super::*;
    /// Pins the relationship to `to_luma8` rather than asserting equality:
    /// exact on greys and single channels, never off by more than one level
    /// anywhere. If a future `image` release drifts further than that, this
    /// fails and the difference gets looked at instead of silently changing
    /// what the adaptive threshold sees.
    #[test]
    fn parallel_luma_tracks_the_reference_conversion_within_one_level() {
        let mut img = image::RgbImage::new(256, 256);
        for (i, p) in img.pixels_mut().enumerate() {
            // Co-prime strides so r, g and b sweep independently.
            *p = image::Rgb([(i % 256) as u8, (i / 256) as u8, (i * 7 % 256) as u8]);
        }
        let dynimg = DynamicImage::ImageRgb8(img);
        let (a, b) = (to_luma8_par(&dynimg), dynimg.to_luma8());
        let worst = a
            .as_raw()
            .iter()
            .zip(b.as_raw())
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 1, "drifted by {worst} grey levels from to_luma8");

        // Greys have no colour weighting to round, so they must be exact -
        // r = g = b = v must map to v, or the image has been shifted wholesale.
        let mut img = image::RgbImage::new(256, 1);
        for x in 0..256u32 {
            img.put_pixel(x, 0, image::Rgb([x as u8, x as u8, x as u8]));
        }
        let dynimg = DynamicImage::ImageRgb8(img);
        let got = to_luma8_par(&dynimg);
        assert_eq!(got.as_raw(), dynimg.to_luma8().as_raw());
        assert!(got
            .as_raw()
            .iter()
            .enumerate()
            .all(|(i, v)| *v as usize == i));
    }

    #[test]
    fn non_rgb_input_falls_back_rather_than_guessing() {
        let img =
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap());
        assert_eq!(to_luma8_par(&img).as_raw(), &vec![1u8, 2, 3, 4]);
    }
}
