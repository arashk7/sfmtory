//! Normalises a declared dataset layout into the tree `dataset::discover`
//! already understands.
//!
//! Some datasets carry capture and camera identity in names that no amount of
//! inspecting the directory shape can decode. The motivating one: a rig of
//! fixed cameras photographing a target that moves between captures, dumped as
//! one directory per capture with one file per camera inside it -
//!
//! ```text
//! images/9/101.jpg     capture 9,  camera 101
//! images/9/102.jpg     capture 9,  camera 102
//! images/16/101.jpg    capture 16, camera 101
//! ```
//!
//!
//! That is shaped exactly like `images/<camera>/<shot>.jpg`, a layout
//! discovery already supports and reads the other way round. Rather than teach
//! every stage a special case, the `[layout]` declaration in `sfm.toml` is
//! resolved here into a symlink tree in the canonical `capture_<n>/cam<n>/`
//! form. Nothing downstream changes, the mapping is inspectable on disk before
//! a long run, and symlinks mean no image is copied.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::dataset::is_image_file;
use crate::project::{IdSource, LayoutConfig};

/// One raw image and the identity the layout gives it.
#[derive(Debug, Clone)]
pub struct Placed {
    pub source: PathBuf,
    pub capture_id: u32,
    pub camera_id: u32,
    /// Path of the symlink to create, relative to the link tree's root.
    pub link: PathBuf,
}

#[derive(Debug, Default)]
pub struct Plan {
    pub placed: Vec<Placed>,
    pub captures: BTreeSet<u32>,
    pub cameras: BTreeSet<u32>,
    /// Cameras absent from at least one capture, with the captures they are
    /// missing from. A rig dataset is usually meant to be complete, and a
    /// camera that silently drops out of some captures weakens exactly the
    /// multi-capture redundancy the layout exists to provide.
    pub gaps: BTreeMap<u32, Vec<u32>>,
}

/// Reads `source` and works out where every image belongs. Creates nothing.
pub fn plan(source: &Path, cfg: &LayoutConfig) -> Result<Plan> {
    if !source.is_dir() {
        bail!("layout source directory does not exist: {}", source.display());
    }
    let files = collect_images(source)?;
    if files.is_empty() {
        bail!("no images found under {}", source.display());
    }

    // Keys are resolved globally, not per directory, so one camera keeps one
    // id across every capture it appears in. Assigning per directory - which
    // is what `dataset::ids_for` does, correctly, for its own layouts - would
    // misalign the moment a camera is missing from one capture, and silently:
    // the ids would still be dense and unique, just attached to the wrong
    // cameras.
    let capture_keys: Vec<String> = files.iter().map(|f| key(f, cfg.capture)).collect();
    let camera_keys: Vec<String> = files.iter().map(|f| key(f, cfg.camera)).collect();
    let capture_ids = assign_ids(&capture_keys);
    let camera_ids = assign_ids(&camera_keys);

    let mut placed = Vec::with_capacity(files.len());
    let mut seen: BTreeMap<(u32, u32), PathBuf> = BTreeMap::new();
    for (i, f) in files.iter().enumerate() {
        let capture_id = capture_ids[&capture_keys[i]];
        let camera_id = camera_ids[&camera_keys[i]];
        let file_name = f
            .path
            .file_name()
            .map(|s| s.to_os_string())
            .unwrap_or_default();
        // A collision means two raw images claim one (capture, camera) slot.
        // One would overwrite the other, so refuse rather than lose data.
        if let Some(prev) = seen.get(&(capture_id, camera_id)) {
            bail!(
                "layout maps two images to capture {capture_id}, camera {camera_id}:\n  \
                 {}\n  {}\nCheck the [layout] capture/camera settings in sfm.toml.",
                prev.display(),
                f.path.display()
            );
        }
        seen.insert((capture_id, camera_id), f.path.clone());
        placed.push(Placed {
            source: f.path.clone(),
            capture_id,
            camera_id,
            link: PathBuf::from(format!("capture_{capture_id:03}"))
                .join(format!("cam{camera_id:03}"))
                .join(file_name),
        });
    }

    let captures: BTreeSet<u32> = placed.iter().map(|p| p.capture_id).collect();
    let cameras: BTreeSet<u32> = placed.iter().map(|p| p.camera_id).collect();
    let mut gaps: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for cam in &cameras {
        let missing: Vec<u32> = captures
            .iter()
            .filter(|c| !seen.contains_key(&(**c, *cam)))
            .copied()
            .collect();
        if !missing.is_empty() {
            gaps.insert(*cam, missing);
        }
    }

    Ok(Plan {
        placed,
        captures,
        cameras,
        gaps,
    })
}

/// Builds the symlink tree. Returns the number of links created.
pub fn apply(plan: &Plan, target: &Path, force: bool) -> Result<usize> {
    if target.exists() {
        if !force {
            bail!(
                "{} already exists; pass --force to rebuild it",
                target.display()
            );
        }
        // Only ever removes the tree this command owns, never the raw images:
        // `target` is under `cache/`, and the links inside point outward.
        std::fs::remove_dir_all(target)
            .with_context(|| format!("removing {}", target.display()))?;
    }
    for p in &plan.placed {
        let dest = target.join(&p.link);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        // Absolute targets, so the tree keeps working regardless of where it
        // is read from or how deep the link sits.
        let source = p
            .source
            .canonicalize()
            .with_context(|| format!("resolving {}", p.source.display()))?;
        symlink(&source, &dest)
            .with_context(|| format!("linking {} -> {}", dest.display(), source.display()))?;
    }
    Ok(plan.placed.len())
}

#[cfg(unix)]
fn symlink(source: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, dest)
}

#[cfg(windows)]
fn symlink(source: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(source, dest)
}

struct RawImage {
    path: PathBuf,
    /// Name of the directory immediately containing the file, empty when the
    /// file sits at the root of the source tree.
    dir: String,
    stem: String,
}

fn collect_images(source: &Path) -> Result<Vec<RawImage>> {
    let mut out = Vec::new();
    walk(source, source, &mut out)?;
    // Sorted so ids and reports are stable between runs.
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<RawImage>) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out)?;
        } else if is_image_file(&path) {
            out.push(RawImage {
                dir: path
                    .parent()
                    .filter(|p| *p != root)
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                stem: path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path,
            });
        }
    }
    Ok(())
}

fn key(f: &RawImage, from: IdSource) -> String {
    match from {
        IdSource::Dir => f.dir.clone(),
        IdSource::Stem => f.stem.clone(),
        IdSource::None => String::new(),
    }
}

/// Maps each distinct key to an id, preferring the number the name already
/// carries so the user's own numbering survives - the same rule
/// `dataset::ids_for` applies to directories, but over the whole dataset.
fn assign_ids(keys: &[String]) -> BTreeMap<String, u32> {
    let distinct: BTreeSet<&String> = keys.iter().collect();
    let parsed: Vec<Option<u32>> = distinct.iter().map(|k| trailing_number(k)).collect();
    let unique = {
        let mut v: Vec<u32> = parsed.iter().flatten().copied().collect();
        v.sort_unstable();
        let before = v.len();
        v.dedup();
        v.len() == before
    };
    if parsed.iter().all(Option::is_some) && unique {
        distinct
            .into_iter()
            .zip(parsed)
            .map(|(k, n)| (k.clone(), n.unwrap()))
            .collect()
    } else {
        distinct
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as u32))
            .collect()
    }
}

/// Trailing run of digits, e.g. `cam007` -> 7. Mirrors `dataset`'s own rule.
fn trailing_number(name: &str) -> Option<u32> {
    let digits: String = name
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sfmtory_layout_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn cfg(capture: IdSource, camera: IdSource) -> LayoutConfig {
        LayoutConfig {
            source: None,
            capture,
            camera,
        }
    }

    /// The `p44` shape: capture directories holding one file per camera.
    #[test]
    fn capture_dirs_of_per_camera_files() {
        let root = tmp("captures");
        for cap in ["9", "16", "23"] {
            for cam in ["101", "7", "58"] {
                touch(&root.join(cap).join(format!("{cam}.jpg")));
            }
        }
        let p = plan(&root, &cfg(IdSource::Dir, IdSource::Stem)).unwrap();
        assert_eq!(p.placed.len(), 9);
        assert_eq!(p.captures, [9, 16, 23].into_iter().collect());
        assert_eq!(p.cameras, [7, 58, 101].into_iter().collect());
        assert!(p.gaps.is_empty());
        // The ids follow the names, and the link path is the canonical form
        // `dataset::discover` reads as captures-and-cameras.
        let one = p
            .placed
            .iter()
            .find(|x| x.source.ends_with("16/101.jpg"))
            .unwrap();
        assert_eq!(one.capture_id, 16);
        assert_eq!(one.camera_id, 101);
        assert_eq!(one.link, PathBuf::from("capture_016/cam101/101.jpg"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// One camera must keep one id even when it is missing from a capture -
    /// the failure a per-directory assignment would introduce silently.
    #[test]
    fn camera_ids_stay_aligned_across_captures_with_a_gap() {
        let root = tmp("gap");
        touch(&root.join("1").join("alpha.jpg"));
        touch(&root.join("1").join("beta.jpg"));
        touch(&root.join("2").join("beta.jpg")); // alpha missing here
        let p = plan(&root, &cfg(IdSource::Dir, IdSource::Stem)).unwrap();
        let id_of = |cap: u32, stem: &str| {
            p.placed
                .iter()
                .find(|x| x.capture_id == cap && x.source.ends_with(format!("{stem}.jpg")))
                .unwrap()
                .camera_id
        };
        assert_eq!(id_of(1, "beta"), id_of(2, "beta"), "beta changed id");
        assert_ne!(id_of(1, "alpha"), id_of(1, "beta"));
        // And the gap is reported rather than passed over.
        assert_eq!(p.gaps.len(), 1);
        let (cam, missing) = p.gaps.iter().next().unwrap();
        assert_eq!(*cam, id_of(1, "alpha"));
        assert_eq!(missing, &vec![2]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn non_numeric_names_fall_back_to_sorted_ordinals() {
        let root = tmp("named");
        for cap in ["morning", "evening"] {
            for cam in ["left", "right"] {
                touch(&root.join(cap).join(format!("{cam}.png")));
            }
        }
        let p = plan(&root, &cfg(IdSource::Dir, IdSource::Stem)).unwrap();
        assert_eq!(p.captures, [0, 1].into_iter().collect());
        assert_eq!(p.cameras, [0, 1].into_iter().collect());
        assert_eq!(p.placed.len(), 4);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_transposed_layout_also_works() {
        // Camera directories holding one file per capture.
        let root = tmp("transposed");
        for cam in ["cam1", "cam2"] {
            for cap in ["5", "6"] {
                touch(&root.join(cam).join(format!("{cap}.jpg")));
            }
        }
        let p = plan(&root, &cfg(IdSource::Stem, IdSource::Dir)).unwrap();
        assert_eq!(p.captures, [5, 6].into_iter().collect());
        assert_eq!(p.cameras, [1, 2].into_iter().collect());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn colliding_images_are_refused_rather_than_overwritten() {
        // Two files in one capture that both resolve to the same camera,
        // because the extension is what differs and the stem is the key.
        let root = tmp("collide");
        touch(&root.join("1").join("7.jpg"));
        touch(&root.join("1").join("7.png"));
        let err = plan(&root, &cfg(IdSource::Dir, IdSource::Stem)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("two images"), "{msg}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn apply_creates_a_tree_discovery_reads_back_as_captures_and_cameras() {
        let root = tmp("apply");
        let src = root.join("raw");
        for cap in ["9", "16"] {
            for cam in ["101", "102"] {
                touch(&src.join(cap).join(format!("{cam}.jpg")));
            }
        }
        let p = plan(&src, &cfg(IdSource::Dir, IdSource::Stem)).unwrap();
        let target = root.join("linked");
        assert_eq!(apply(&p, &target, false).unwrap(), 4);

        let (imgs, layout) = crate::dataset::discover(&target).unwrap();
        assert_eq!(layout, crate::dataset::Layout::CapturesAndCameras);
        assert_eq!(imgs.len(), 4);
        let caps: BTreeSet<u32> = imgs.iter().map(|i| i.capture_id).collect();
        let cams: BTreeSet<u32> = imgs.iter().map(|i| i.camera_id).collect();
        assert_eq!(caps, [9, 16].into_iter().collect());
        assert_eq!(cams, [101, 102].into_iter().collect());
        // One image per (capture, camera), so every slot index is zero and
        // `--merge-multicaps` keys on the camera alone.
        assert!(imgs.iter().all(|i| i.image_index == 0));

        // Rebuilding without --force must refuse rather than silently replace.
        assert!(apply(&p, &target, false).is_err());
        assert_eq!(apply(&p, &target, true).unwrap(), 4);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
