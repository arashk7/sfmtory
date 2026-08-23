//! Project directory layout shared by every `sfm` subcommand (see PLAN.md §2):
//! `sfm.toml`, `database.sqlite`, `sparse/0/`, `export/`, `logs/`. Keeping this
//! in one place is what lets each pipeline stage run as an independent CLI
//! invocation while still finding the previous stage's output.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub images_dir: PathBuf,
    #[serde(default)]
    pub detector: Option<String>,
    #[serde(default)]
    pub pairing: Option<String>,
    #[serde(default)]
    pub matcher: Option<String>,
    #[serde(default)]
    pub pipeline: Option<String>,
}

pub struct Project {
    pub root: PathBuf,
    pub config: ProjectConfig,
}

impl Project {
    pub fn config_path(root: &Path) -> PathBuf {
        root.join("sfm.toml")
    }

    pub fn database_path(&self) -> PathBuf {
        self.root.join("database.sqlite")
    }

    pub fn sparse_dir(&self) -> PathBuf {
        self.root.join("sparse").join("0")
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
        fs::create_dir_all(root.join("sparse").join("0"))?;
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
        };
        let toml_str = toml::to_string_pretty(&config)?;
        fs::write(Self::config_path(root), toml_str)?;

        Ok(Project {
            root: root.to_path_buf(),
            config,
        })
    }

    /// Load an existing project.
    pub fn open(root: &Path) -> Result<Self> {
        let config_path = Self::config_path(root);
        let content = fs::read_to_string(&config_path).with_context(|| {
            format!(
                "reading {} (did you run `sfm project new`?)",
                config_path.display()
            )
        })?;
        let config: ProjectConfig = toml::from_str(&content)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        Ok(Project {
            root: root.to_path_buf(),
            config,
        })
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
