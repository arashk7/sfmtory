//! The "Start here" view: what to do next, and whether the result can mean
//! anything when it arrives.
//!
//! The viewer used to open on the 3D scene with a one-line hint in the toolbar.
//! That is the wrong first screen for two reasons. On a fresh project the scene
//! is empty, so the view carries no information and the hint competes with a
//! dozen buttons for attention. And on a *finished* project the scene looks
//! equally plausible whether or not the cameras were ever calibrated - a
//! reconstruction that kept its initial field-of-view guess renders a tidy
//! point cloud and four confident-looking cameras, and nothing on screen says
//! the intrinsics are a guess.
//!
//! So this view answers two questions in order: what is the next command, and
//! will its output be a calibration or a picture of one. The second question is
//! the one that cost real debugging time here, and every check below exists
//! because it caught something on a real dataset.

use std::collections::BTreeMap;
use std::path::Path;

use crate::project::Project;

/// A single readiness check, phrased as something the user can act on.
pub struct Check {
    pub ok: bool,
    /// Short label, e.g. "camera models".
    pub what: &'static str,
    /// What is currently true.
    pub state: String,
    /// What to do about it, when there is something to do.
    pub fix: Option<String>,
    /// The command that does it, ready to spawn.
    pub command: Option<Vec<String>>,
}

pub struct Readiness {
    pub num_images: usize,
    pub num_captures: usize,
    pub num_cameras: usize,
    pub images_per_camera: BTreeMap<u32, usize>,
    pub merged: bool,
    pub declared_models: Vec<(String, String)>,
    pub checks: Vec<Check>,
}

impl Readiness {
    pub fn read(project_dir: &Path) -> Self {
        let mut r = Readiness {
            num_images: 0,
            num_captures: 0,
            num_cameras: 0,
            images_per_camera: BTreeMap::new(),
            merged: false,
            declared_models: Vec::new(),
            checks: Vec::new(),
        };
        let Ok(project) = Project::open(project_dir) else {
            return r;
        };

        // What the dataset contains, before any stage has run - so the advice
        // is available at the point it is most useful.
        if let Ok(dir) = project.require_images_dir() {
            if let Ok((discovered, _)) = crate::dataset::discover(&dir) {
                r.num_images = discovered.len();
                r.num_captures = discovered
                    .iter()
                    .map(|d| d.capture_id)
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                r.num_cameras = discovered
                    .iter()
                    .map(|d| d.camera_id)
                    .collect::<std::collections::HashSet<_>>()
                    .len();
            }
        }
        r.declared_models = project
            .config
            .cameras
            .iter()
            .map(|c| {
                (
                    c.images.clone(),
                    c.model.clone().unwrap_or_else(|| "SIMPLE_RADIAL".into()),
                )
            })
            .collect();

        if let Ok(db) = crate::db::Database::open(&project.database_path()) {
            if let Ok(images) = db.list_images() {
                for (_, camera_id, ..) in &images {
                    *r.images_per_camera.entry(*camera_id).or_insert(0) += 1;
                }
                // Merging pools every capture of a camera into one feature set,
                // so a multi-capture dataset that extracted to one image per
                // camera was merged.
                r.merged = r.num_captures > 1
                    && !images.is_empty()
                    && images.len() == r.images_per_camera.len();
            }
        }

        r.checks = r.build_checks();
        r
    }

    fn build_checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();

        // 1. Are the lenses declared? An undeclared camera is SIMPLE_RADIAL at
        // a field-of-view guess, which is silently wrong for a fisheye.
        checks.push(if self.declared_models.is_empty() {
            Check {
                ok: false,
                what: "camera models",
                state: format!(
                    "none declared - all {} camera(s) will be SIMPLE_RADIAL at a guessed focal",
                    self.num_cameras.max(1)
                ),
                fix: Some(
                    "SIMPLE_RADIAL cannot represent a fisheye lens, and a fisheye fitted as one \
                     is placed at the wrong depth. Declare the models, or search for them."
                        .into(),
                ),
                command: Some(vec!["select-model".into(), "--rebuild".into()]),
            }
        } else {
            Check {
                ok: true,
                what: "camera models",
                state: self
                    .declared_models
                    .iter()
                    .map(|(g, m)| format!("{g} -> {m}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                fix: None,
                command: None,
            }
        });

        // 2. Multi-capture datasets need merging, or each capture becomes its
        // own island in the match graph.
        if self.num_captures > 1 {
            checks.push(if self.merged {
                Check {
                    ok: true,
                    what: "captures",
                    state: format!("{} captures merged per camera", self.num_captures),
                    fix: None,
                    command: None,
                }
            } else {
                Check {
                    ok: false,
                    what: "captures",
                    state: format!("{} captures, not merged", self.num_captures),
                    fix: Some(
                        "A fiducial that was moved between captures is a different 3D point in \
                         each one, so captures do not match each other and each becomes its own \
                         island - only one of them will reconstruct. Merge them per camera."
                            .into(),
                    ),
                    command: Some(vec![
                        "feature".into(),
                        "--detector".into(),
                        "aruco".into(),
                        "--merge-multicaps".into(),
                    ]),
                }
            });
        }

        // 3. The one that decides whether "calibration" happened at all.
        let min_images = self.images_per_camera.values().copied().min().unwrap_or(0);
        if !self.images_per_camera.is_empty() {
            let need = sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS;
            checks.push(if min_images >= need {
                Check {
                    ok: true,
                    what: "focal length",
                    state: format!("{min_images} image(s) per camera - refinable"),
                    fix: None,
                    command: None,
                }
            } else {
                Check {
                    ok: false,
                    what: "focal length",
                    state: format!(
                        "{min_images} image(s) on the thinnest camera - needs {need} to refine"
                    ),
                    fix: Some(
                        "Moving the focal and moving the camera toward or away from the scene \
                         produce almost the same picture, so one view cannot separate them. The \
                         focal will stay at its initial guess and the reconstruction will still \
                         look plausible. Either give each camera views of differently-placed \
                         targets, or supply known intrinsics with refine = false."
                            .into(),
                    ),
                    command: None,
                }
            });
        }

        checks
    }

    /// Whether anything below is worth the user's attention before they trust
    /// the numbers.
    pub fn all_ok(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Readiness {
        Readiness {
            num_images: 4,
            num_captures: 1,
            num_cameras: 4,
            images_per_camera: (1..=4).map(|i| (i, 1)).collect(),
            merged: false,
            declared_models: Vec::new(),
            checks: Vec::new(),
        }
    }

    #[test]
    fn undeclared_models_are_flagged_with_a_command_to_fix_them() {
        let r = base();
        let checks = r.build_checks();
        let c = checks.iter().find(|c| c.what == "camera models").unwrap();
        assert!(!c.ok);
        // The point of the check is the next command, not the complaint.
        assert_eq!(
            c.command.as_deref(),
            Some(["select-model".to_string(), "--rebuild".to_string()].as_slice())
        );
    }

    #[test]
    fn declared_models_pass_and_are_listed() {
        let mut r = base();
        r.declared_models = vec![("*cam001*".into(), "OPENCV_FISHEYE".into())];
        let checks = r.build_checks();
        let c = checks.iter().find(|c| c.what == "camera models").unwrap();
        assert!(c.ok);
        assert!(c.state.contains("OPENCV_FISHEYE"));
    }

    #[test]
    fn a_single_capture_project_is_not_asked_to_merge() {
        let r = base();
        assert!(r.build_checks().iter().all(|c| c.what != "captures"));
    }

    #[test]
    fn unmerged_multicapture_is_flagged() {
        let mut r = base();
        r.num_captures = 5;
        let checks = r.build_checks();
        let c = checks.iter().find(|c| c.what == "captures").unwrap();
        assert!(!c.ok);
        assert!(c
            .command
            .as_ref()
            .unwrap()
            .contains(&"--merge-multicaps".to_string()));
    }

    #[test]
    fn one_image_per_camera_fails_the_focal_check_with_no_command() {
        let r = base();
        let checks = r.build_checks();
        let c = checks.iter().find(|c| c.what == "focal length").unwrap();
        assert!(!c.ok);
        // Deliberately no command: no stage can fix this, only different data
        // or known intrinsics can, and offering a button would imply otherwise.
        assert!(c.command.is_none());
        assert!(c.fix.as_ref().unwrap().contains("refine = false"));
    }

    #[test]
    fn enough_images_passes_the_focal_check() {
        let mut r = base();
        r.images_per_camera = (1..=4)
            .map(|i| (i, sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS))
            .collect();
        let checks = r.build_checks();
        assert!(checks.iter().find(|c| c.what == "focal length").unwrap().ok);
    }

    #[test]
    fn all_ok_requires_every_check() {
        let mut r = base();
        r.checks = r.build_checks();
        assert!(!r.all_ok());
        r.declared_models = vec![("*".into(), "RADIAL3".into())];
        r.images_per_camera = (1..=4).map(|i| (i, 8)).collect();
        r.checks = r.build_checks();
        assert!(r.all_ok());
    }
}
