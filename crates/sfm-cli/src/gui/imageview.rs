//! Loads one source photograph so the viewer can draw its features and
//! residuals on top of it.
//!
//! This is the fastest way to separate "detection is fine, matching isn't"
//! from the reverse. A frame densely covered in detected keypoints of which
//! almost none carry a 3D link says the detector did its job and matching or
//! triangulation did not; a frame with barely any keypoints at all says the
//! opposite, and no aggregate statistic distinguishes the two.

use std::path::{Path, PathBuf};

use eframe::egui;

/// Long-edge cap for the decoded texture. A modern phone photo is ~12MP,
/// which is 48MB as RGBA and pointless to upload in full - the overlay is
/// read at screen scale. Keypoint coordinates are rescaled to match.
const MAX_TEXTURE_EDGE: u32 = 2000;

pub struct LoadedImage {
    pub image_id: u32,
    pub texture: egui::TextureHandle,
    /// Size of the *source* image in pixels, which is the space keypoint
    /// coordinates live in.
    pub source_size: [f32; 2],
    pub path: PathBuf,
}

/// Where an image named `name` might be on disk.
///
/// `images_dir` is stored absolute in `sfm.toml`, so a project directory that
/// has been moved or copied (which is normal for a dataset) would otherwise
/// resolve to nothing; falling back to the conventional `<root>/images` keeps
/// the overlay working in that case.
pub fn candidate_paths(root: &Path, images_dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut v = vec![images_dir.join(name)];
    let conventional = root.join("images").join(name);
    if !v.contains(&conventional) {
        v.push(conventional);
    }
    v
}

pub fn load(
    ctx: &egui::Context,
    root: &Path,
    images_dir: &Path,
    image_id: u32,
    name: &str,
) -> Result<LoadedImage, String> {
    let candidates = candidate_paths(root, images_dir, name);
    let path = candidates
        .iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            format!(
                "could not find the source image for \"{name}\"; looked in {}",
                candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" and ")
            )
        })?
        .clone();

    let decoded = image::open(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let rgba = decoded.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let source_size = [w as f32, h as f32];

    let scale = MAX_TEXTURE_EDGE as f32 / w.max(h) as f32;
    let rgba = if scale < 1.0 {
        image::imageops::resize(
            &rgba,
            ((w as f32 * scale) as u32).max(1),
            ((h as f32 * scale) as u32).max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        rgba
    };

    let color = egui::ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        rgba.as_raw(),
    );
    // One texture handle per selected image, replaced on selection change -
    // the viewer never shows two source photos at once, so caching more than
    // one would only hold VRAM.
    let texture = ctx.load_texture(
        format!("source-{image_id}"),
        color,
        egui::TextureOptions::LINEAR,
    );
    Ok(LoadedImage {
        image_id,
        texture,
        source_size,
        path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_the_conventional_images_directory() {
        let root = Path::new("/data/set");
        // A stale absolute path from a project directory that has since moved.
        let stale = Path::new("/elsewhere/old/images");
        let c = candidate_paths(root, stale, "a.jpg");
        assert_eq!(c[0], PathBuf::from("/elsewhere/old/images/a.jpg"));
        assert_eq!(c[1], PathBuf::from("/data/set/images/a.jpg"));
    }

    #[test]
    fn does_not_repeat_the_conventional_path() {
        let root = Path::new("/data/set");
        let c = candidate_paths(root, &root.join("images"), "sub/a.jpg");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], PathBuf::from("/data/set/images/sub/a.jpg"));
    }
}
