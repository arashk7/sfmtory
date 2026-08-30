//! Project directory layout shared by every `sfmtory` subcommand.
//!
//! ```text
//! <project>/
//!   sfm.toml            # optional - defaults apply when absent
//!   images/             # default input location
//!   cache/
//!     project.sqlite    # shared working store (features, matches)
//!     dataset/images/   # `sfmtory dataset link` symlink tree, when [layout] is set
//!     feature/          # `sfmtory feature` output + learned ArUco params
//!     match/            # `sfmtory match` output
//!     map/sparse/0/     # `sfmtory map` output (COLMAP text model)
//!   export/             # default `sfmtory export` destination
//! ```
//!
//! Each stage is an independent process, so the layout - not shared memory -
//! is what carries state between them. The project root defaults to the
//! current directory, and `sfm.toml` is optional: a directory with an
//! `images/` folder in it is already a valid project.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub images_dir: PathBuf,
    #[serde(default)]
    pub detector: Option<String>,
    #[serde(default)]
    pub pairing: Option<String>,
    #[serde(default)]
    pub matcher: Option<String>,
    #[serde(default)]
    pub pipeline: Option<String>,
    /// Explicit camera definitions. When empty (the default), images are
    /// grouped into cameras by image resolution - which is right for the
    /// common single-camera case but silently merges *distinct* physical
    /// cameras that happen to share a resolution into one shared intrinsics
    /// block. Declare cameras here whenever that would be wrong.
    #[serde(default)]
    pub cameras: Vec<CameraConfig>,
    /// Known camera poses, used to initialize (and optionally pin) extrinsics.
    #[serde(default)]
    pub poses: Vec<PoseConfig>,
    /// How to read a dataset whose directory shape `dataset::discover` cannot
    /// infer. Absent for the three layouts it recognises on its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutConfig>,
}

/// Declares where capture and camera identity live in a dataset's own naming,
/// for layouts `dataset::discover` cannot tell apart by shape alone.
///
/// The ambiguous case that motivates this: `images/A/*.jpg` is equally
/// consistent with "A is one camera that took many shots" and "A is one
/// capture in which each file is a different camera". Discovery assumes the
/// former, because it has to assume something; a rig dumping one file per
/// camera per shot needs the latter, and no amount of inspecting the tree can
/// settle which was meant. So it is declared rather than guessed.
///
/// `sfmtory dataset link` reads this and materialises the canonical
/// `capture_<n>/cam<n>/` tree that discovery already understands, so the
/// declaration affects one command rather than threading a special case
/// through every stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Raw image tree, relative to the project root (or absolute). Defaults to
    /// `images_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<PathBuf>,
    /// What each level of the path means, outermost first, ending with the
    /// file itself.
    ///
    /// One entry per level, so the list length is the depth of the tree:
    ///
    /// ```toml
    /// layers = ["capture", "camera"]            # images/9/101.jpg
    /// layers = ["capture", "camera", "image"]   # images/cap0/cam0/0001.jpg
    /// layers = ["camera", "image"]              # images/cam0/0001.jpg
    /// layers = ["image"]                        # images/0001.jpg
    /// ```
    ///
    /// Naming the levels rather than naming the ids ("capture comes from the
    /// directory, camera from the file stem") means one rule covers any depth
    /// instead of one flag per identity, and the declaration reads in the same
    /// order as the path it describes.
    pub layers: Vec<Layer>,
}

/// What one level of a dataset path identifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    /// A capture: one session of a target, moved between sessions.
    Capture,
    /// A physical camera.
    Camera,
    /// A shot within one (capture, camera) - several frames from one camera.
    Image,
    /// Carries no identity; a wrapper or grouping directory to skip over.
    Ignore,
}

impl LayoutConfig {
    /// The raw tree this layout reads from.
    ///
    /// Relative paths resolve against the project root rather than the working
    /// directory, so `--project <elsewhere>` behaves the same as running from
    /// inside the project.
    pub fn source_dir(&self, root: &Path, images_dir: &Path) -> PathBuf {
        let p = self.source.as_deref().unwrap_or(images_dir);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            root.join(p)
        }
    }
}

/// One declared physical camera: which images belong to it, and optionally
/// its known calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Identifies this camera in messages; also the tie-break for ordering.
    pub name: String,
    /// Glob matched against each image's *file name* (not full path).
    /// Supports `*` (any run of characters) and `?` (one character).
    pub images: String,
    /// COLMAP model name, e.g. `SIMPLE_RADIAL`, `PINHOLE`, `OPENCV`.
    /// Defaults to `SIMPLE_RADIAL` when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Known intrinsics in that model's parameter order. When omitted, the
    /// usual focal-length guess from image dimensions is used instead.
    #[serde(default)]
    pub params: Option<Vec<f64>>,
    /// Whether bundle adjustment may refine these intrinsics. Defaults to
    /// `true`; set `false` for a camera you have calibrated offline and want
    /// held exactly.
    #[serde(default)]
    pub refine: Option<bool>,
}

/// A known world-to-camera pose for one image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseConfig {
    /// Image file name this pose belongs to.
    pub image: String,
    /// World-to-camera rotation as a quaternion, `[w, x, y, z]`.
    pub quaternion: [f64; 4],
    /// World-to-camera translation, `t = -R * camera_center`.
    pub translation: [f64; 3],
    /// Whether bundle adjustment may move this pose. Defaults to `false`
    /// (the pose is an initialization only); set `true` to pin it, e.g. for
    /// a rig whose extrinsics were measured separately.
    #[serde(default)]
    pub fixed: Option<bool>,
}

/// Minimal glob matcher supporting `*` and `?`, so declaring cameras doesn't
/// pull in a glob dependency for one line of matching.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    fn go(p: &[u8], n: &[u8]) -> bool {
        match p.first() {
            None => n.is_empty(),
            Some(b'*') => {
                // Try consuming zero or more characters against `*`.
                (0..=n.len()).any(|k| go(&p[1..], &n[k..]))
            }
            Some(&c) => !n.is_empty() && (c == b'?' || c == n[0]) && go(&p[1..], &n[1..]),
        }
    }
    go(pattern.as_bytes(), name.as_bytes())
}

pub struct Project {
    pub root: PathBuf,
    pub config: ProjectConfig,
}

impl Project {
    pub fn config_path(root: &Path) -> PathBuf {
        root.join("sfm.toml")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// Per-stage output directory, named after the stage that writes it.
    pub fn stage_dir(&self, stage: &str) -> PathBuf {
        self.cache_dir().join(stage)
    }

    /// Shared working store. Lives beside the per-stage directories rather
    /// than inside one of them because more than one stage writes to it -
    /// `feature` creates the images/cameras/features, `match` adds two-view
    /// geometries - and filing it under whichever stage happened to create it
    /// would misrepresent that.
    pub fn database_path(&self) -> PathBuf {
        self.cache_dir().join("project.sqlite")
    }

    pub fn sparse_dir(&self) -> PathBuf {
        self.stage_dir("map").join("sparse").join("0")
    }

    /// Where `sfmtory dataset link` writes its normalised symlink tree.
    ///
    /// Under `cache/` because it is derived data that can be regenerated from
    /// the raw tree and the `[layout]` declaration at any time - the same
    /// reason the feature database lives there.
    pub fn linked_images_dir(&self) -> PathBuf {
        self.cache_dir().join("dataset").join("images")
    }

    /// The image tree the pipeline stages should actually read.
    ///
    /// With a `[layout]` declared this is the linked tree, not the raw one:
    /// the raw tree is by definition in a shape discovery would misread, which
    /// is why the declaration exists.
    pub fn effective_images_dir(&self) -> PathBuf {
        if self.config.layout.is_some() {
            self.linked_images_dir()
        } else {
            self.config.images_dir.clone()
        }
    }

    /// Explains that the linked tree has not been built yet, rather than
    /// letting discovery fail with a bare "directory does not exist".
    pub fn require_images_dir(&self) -> Result<PathBuf> {
        let dir = self.effective_images_dir();
        if self.config.layout.is_some() && !dir.is_dir() {
            anyhow::bail!(
                "sfm.toml declares a [layout], but its linked image tree is missing at {}.\n\
                 Run `sfmtory dataset link` to build it.",
                dir.display()
            );
        }
        Ok(dir)
    }

    /// Default input location when `sfm.toml` doesn't say otherwise.
    pub fn default_images_dir(root: &Path) -> PathBuf {
        root.join("images")
    }

    /// Ensures the cache layout exists for one stage and returns its directory.
    pub fn prepare_stage(&self, stage: &str) -> Result<PathBuf> {
        let dir = self.stage_dir(stage);
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir)
    }

    pub fn export_dir(&self) -> PathBuf {
        self.root.join("export")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    /// Create a new project directory with the standard layout.
    pub fn create(root: &Path, images_dir: &Path) -> Result<Self> {
        fs::create_dir_all(root)
            .with_context(|| format!("creating project directory {}", root.display()))?;
        fs::create_dir_all(root.join("cache").join("feature"))?;
        fs::create_dir_all(root.join("cache").join("match"))?;
        fs::create_dir_all(root.join("cache").join("map"))?;
        fs::create_dir_all(root.join("export"))?;
        fs::create_dir_all(root.join("logs"))?;

        let images_dir = images_dir
            .canonicalize()
            .with_context(|| format!("resolving images directory {}", images_dir.display()))?;
        let config = ProjectConfig {
            images_dir,
            detector: None,
            pairing: None,
            matcher: None,
            pipeline: None,
            cameras: Vec::new(),
            poses: Vec::new(),
            layout: None,
        };
        let toml_str = toml::to_string_pretty(&config)?;
        fs::write(Self::config_path(root), toml_str)?;

        Ok(Project {
            root: root.to_path_buf(),
            config,
        })
    }

    /// Open a project rooted at `root`. `sfm.toml` is optional: without it the
    /// project still works, using `<root>/images` as the input directory. That
    /// keeps the common case - cd into a dataset directory and run the
    /// pipeline - free of setup ceremony.
    pub fn open(root: &Path) -> Result<Self> {
        let config_path = Self::config_path(root);
        let config = match fs::read_to_string(&config_path) {
            Ok(content) => {
                let mut cfg: ProjectConfig = toml::from_str(&content)
                    .with_context(|| format!("parsing {}", config_path.display()))?;
                if cfg.images_dir.as_os_str().is_empty() {
                    cfg.images_dir = Self::default_images_dir(root);
                }
                cfg
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => ProjectConfig {
                images_dir: Self::default_images_dir(root),
                detector: None,
                pairing: None,
                matcher: None,
                pipeline: None,
                cameras: Vec::new(),
                poses: Vec::new(),
                layout: None,
            },
            Err(e) => return Err(e).with_context(|| format!("reading {}", config_path.display())),
        };
        Ok(Project {
            root: root.to_path_buf(),
            config,
        })
    }

    /// Path of the ArUco parameters learned by `sfmtory feature --find-params`.
    pub fn aruco_params_path(&self) -> PathBuf {
        self.stage_dir("feature").join("aruco_params.toml")
    }

    /// Append a JSON result record for one pipeline stage to `logs/`, so every
    /// step's output is on disk regardless of how it was invoked. File name is
    /// `<stage>_<unix_millis>.json`.
    pub fn record_log(&self, stage: &str, payload: &serde_json::Value) -> Result<PathBuf> {
        fs::create_dir_all(self.logs_dir())?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = self.logs_dir().join(format!("{stage}_{millis}.json"));
        fs::write(&path, serde_json::to_string_pretty(payload)?)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn glob_matches_prefix_suffix_and_single_char() {
        assert!(glob_match("left_*", "left_0001.png"));
        assert!(!glob_match("left_*", "right_0001.png"));
        assert!(glob_match("*.png", "cam2_0007.png"));
        assert!(glob_match("cam?_*", "cam3_0001.png"));
        assert!(!glob_match("cam?_*", "cam31_0001.png"));
        assert!(glob_match("*", "anything.jpg"));
        assert!(glob_match("exact.png", "exact.png"));
        assert!(!glob_match("exact.png", "exact.jpg"));
    }
}
