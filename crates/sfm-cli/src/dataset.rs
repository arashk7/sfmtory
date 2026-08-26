//! Input-image discovery.
//!
//! A dataset is laid out in one of three ways, and which one it is has to be
//! inferred rather than declared, because the same pipeline serves all of
//! them:
//!
//! ```text
//! images/capture_000/cam000/image.jpg   # captures x cameras
//! images/cam000/image.jpg               # cameras only (one implicit capture)
//! images/cam000_image.jpg               # flat files
//! ```
//!
//! The distinction matters well beyond bookkeeping. `capture_id` is what keeps
//! a fiducial marker that was *physically moved between captures* from being
//! matched to itself across them (see `sfm-features::aruco`), and
//! `(camera_id, image_index)` is the key `--merge-multicaps` accumulates
//! along.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

pub fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct DiscoveredImage {
    pub path: PathBuf,
    pub capture_id: u32,
    pub camera_id: u32,
    /// Position of this image within its camera directory, so that a camera
    /// holding several shots per capture merges shot-for-shot rather than
    /// pooling them all together.
    pub image_index: u32,
    /// Slash-separated path relative to the images root. Unique, stable, and
    /// what gets stored as the image name (COLMAP does the same).
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    CapturesAndCameras,
    CamerasOnly,
    Flat,
}

/// Trailing run of digits in a directory or file name, e.g. `cam007` -> 7,
/// `capture_012` -> 12. Used so ids follow the names the user chose rather
/// than discovery order.
fn trailing_number(name: &str) -> Option<u32> {
    let digits: String = name
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Leading `cam<N>` (or `camera<N>`) prefix of a flat file name, e.g.
/// `cam003_image.jpg` -> 3.
fn leading_camera_number(file_stem: &str) -> Option<u32> {
    let lower = file_stem.to_ascii_lowercase();
    let rest = lower
        .strip_prefix("camera")
        .or_else(|| lower.strip_prefix("cam"))?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn sorted_dirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    Ok(v)
}

fn sorted_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_image_file(p))
        .collect();
    v.sort();
    Ok(v)
}

fn dir_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn rel_name(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Assigns ids from directory names where they carry a number, falling back to
/// sorted ordinal position where they don't - so `cam000/ cam001/` and
/// `left/ right/` both work, the former keeping the user's numbering.
fn ids_for(dirs: &[PathBuf]) -> Vec<u32> {
    let parsed: Vec<Option<u32>> = dirs.iter().map(|d| trailing_number(&dir_name(d))).collect();
    let all_numbered = parsed.iter().all(|p| p.is_some());
    let unique = {
        let mut v: Vec<u32> = parsed.iter().flatten().copied().collect();
        v.sort_unstable();
        let before = v.len();
        v.dedup();
        v.len() == before
    };
    if all_numbered && unique {
        parsed.into_iter().map(|p| p.unwrap()).collect()
    } else {
        (0..dirs.len() as u32).collect()
    }
}

/// Detects the layout and enumerates every image with its capture/camera ids.
pub fn discover(images_root: &Path) -> Result<(Vec<DiscoveredImage>, Layout)> {
    if !images_root.is_dir() {
        bail!("images directory does not exist: {}", images_root.display());
    }

    let top_dirs = sorted_dirs(images_root)?;
    let top_images = sorted_images(images_root)?;

    // Captures-and-cameras when the second level also holds directories;
    // cameras-only when the second level holds images.
    let nested = !top_dirs.is_empty()
        && top_dirs.iter().any(|d| {
            sorted_dirs(d)
                .map(|sub| sub.iter().any(|s| sorted_images(s).map(|i| !i.is_empty()).unwrap_or(false)))
                .unwrap_or(false)
        });

    let mut out = Vec::new();

    if nested {
        let capture_ids = ids_for(&top_dirs);
        for (cap_pos, cap_dir) in top_dirs.iter().enumerate() {
            let cam_dirs = sorted_dirs(cap_dir)?;
            let cam_ids = ids_for(&cam_dirs);
            for (cam_pos, cam_dir) in cam_dirs.iter().enumerate() {
                for (idx, img) in sorted_images(cam_dir)?.into_iter().enumerate() {
                    out.push(DiscoveredImage {
                        name: rel_name(images_root, &img),
                        path: img,
                        capture_id: capture_ids[cap_pos],
                        camera_id: cam_ids[cam_pos],
                        image_index: idx as u32,
                    });
                }
            }
        }
        if out.is_empty() {
            bail!("no images found under {}", images_root.display());
        }
        return Ok((out, Layout::CapturesAndCameras));
    }

    if !top_dirs.is_empty() {
        let cam_ids = ids_for(&top_dirs);
        for (cam_pos, cam_dir) in top_dirs.iter().enumerate() {
            for (idx, img) in sorted_images(cam_dir)?.into_iter().enumerate() {
                out.push(DiscoveredImage {
                    name: rel_name(images_root, &img),
                    path: img,
                    capture_id: 0,
                    camera_id: cam_ids[cam_pos],
                    image_index: idx as u32,
                });
            }
        }
        if !out.is_empty() {
            return Ok((out, Layout::CamerasOnly));
        }
    }

    // Flat: a `cam<N>` prefix names the camera if present, otherwise every
    // image is treated as coming from one camera - the ordinary
    // single-camera case.
    let mut per_camera_count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for img in &top_images {
        let stem = img
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let camera_id = leading_camera_number(&stem).unwrap_or(0);
        let slot = per_camera_count.entry(camera_id).or_insert(0);
        out.push(DiscoveredImage {
            name: rel_name(images_root, img),
            path: img.clone(),
            capture_id: 0,
            camera_id,
            image_index: *slot,
        });
        *slot += 1;
    }
    if out.is_empty() {
        bail!("no images found in {}", images_root.display());
    }
    Ok((out, Layout::Flat))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sfmtory_ds_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn discovers_captures_and_cameras() {
        let root = tmp("nested");
        for cap in 0..3 {
            for cam in 0..2 {
                touch(&root.join(format!("capture_{cap:03}/cam{cam:03}/image.jpg")));
            }
        }
        let (imgs, layout) = discover(&root).unwrap();
        assert_eq!(layout, Layout::CapturesAndCameras);
        assert_eq!(imgs.len(), 6);
        let c1 = imgs.iter().find(|i| i.name.contains("capture_001/cam001")).unwrap();
        assert_eq!((c1.capture_id, c1.camera_id, c1.image_index), (1, 1, 0));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn discovers_cameras_only_as_single_capture() {
        let root = tmp("cams");
        touch(&root.join("cam000/a.png"));
        touch(&root.join("cam000/b.png"));
        touch(&root.join("cam007/a.png"));
        let (imgs, layout) = discover(&root).unwrap();
        assert_eq!(layout, Layout::CamerasOnly);
        assert_eq!(imgs.len(), 3);
        assert!(imgs.iter().all(|i| i.capture_id == 0));
        // Second image in cam000 gets index 1, so merging pairs shot-for-shot.
        let b = imgs.iter().find(|i| i.name.ends_with("cam000/b.png")).unwrap();
        assert_eq!((b.camera_id, b.image_index), (0, 1));
        // Camera id follows the directory name, not discovery order.
        assert!(imgs.iter().any(|i| i.camera_id == 7));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn discovers_flat_with_camera_prefix() {
        let root = tmp("flat");
        touch(&root.join("cam000_image.jpg"));
        touch(&root.join("cam001_image.jpeg"));
        touch(&root.join("cam001_other.png"));
        let (imgs, layout) = discover(&root).unwrap();
        assert_eq!(layout, Layout::Flat);
        assert_eq!(imgs.len(), 3);
        assert_eq!(imgs.iter().filter(|i| i.camera_id == 1).count(), 2);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn flat_without_prefix_is_one_camera() {
        let root = tmp("plain");
        touch(&root.join("a.jpg"));
        touch(&root.join("b.jpg"));
        let (imgs, layout) = discover(&root).unwrap();
        assert_eq!(layout, Layout::Flat);
        assert!(imgs.iter().all(|i| i.camera_id == 0 && i.capture_id == 0));
        assert_eq!(imgs[1].image_index, 1);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn non_numeric_directories_fall_back_to_ordinals() {
        let root = tmp("named");
        touch(&root.join("left/a.jpg"));
        touch(&root.join("right/a.jpg"));
        let (imgs, _) = discover(&root).unwrap();
        let mut ids: Vec<u32> = imgs.iter().map(|i| i.camera_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ignores_non_image_files() {
        let root = tmp("mixed");
        touch(&root.join("a.jpg"));
        touch(&root.join("notes.txt"));
        let (imgs, _) = discover(&root).unwrap();
        assert_eq!(imgs.len(), 1);
        std::fs::remove_dir_all(&root).ok();
    }
}
