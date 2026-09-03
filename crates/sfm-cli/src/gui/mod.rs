//! `sfmtory gui` - a viewer and front-end over the same pipeline the CLI runs.
//!
//! The CLI remains the primary interface: every button here shells out to the
//! same subcommand you would type, against the same project directory, so
//! nothing is reachable through the GUI that is not reachable without it.
//!
//! Beyond the 3D view, the panels here exist to answer questions the pipeline
//! already has the data for and never says out loud - why a focal length did
//! not refine, whether a planar capture was shot from enough angles, whether a
//! 0.3px mean error is uniform or hides a 572px outlier, and whether the
//! images that failed to register did so because they were never connected to
//! the rest of the match graph at all.

mod aruco;
mod graph;
mod imageview;
mod pipeline;
mod scene;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use eframe::egui;
use sfm_core::Reconstruction;

use crate::diagnostics::{self, CoverageVerdict, Diagnostics, PlaneDiag, TILT_BANDS};
use crate::project::Project;
use graph::MatchGraph;
use imageview::LoadedImage;
mod start;

use pipeline::RunState;
use scene::{OrbitCamera, PlaneOverlay, RenderOptions, Scene};

pub fn launch(project_dir: PathBuf, view: View) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("sfmtory"),
        ..Default::default()
    };
    eframe::run_native(
        "sfmtory",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(project_dir, view)))),
    )
    .map_err(|e| anyhow::anyhow!("could not start the viewer: {e}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// What to do next, and whether the answer will mean anything.
    Start,
    Scene3d,
    Image,
    Graph,
    Coverage,
    Aruco,
}

impl From<crate::GuiViewArg> for View {
    fn from(v: crate::GuiViewArg) -> Self {
        match v {
            crate::GuiViewArg::Start => View::Start,
            crate::GuiViewArg::Scene => View::Scene3d,
            crate::GuiViewArg::Image => View::Image,
            crate::GuiViewArg::Graph => View::Graph,
            crate::GuiViewArg::Coverage => View::Coverage,
            crate::GuiViewArg::Aruco => View::Aruco,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    /// Each point in its own photometric colour, as reconstructed.
    Photometric,
    /// Each point ramped by its mean reprojection residual.
    Residual,
}

/// The match graph is built by reading every verified pair back out of the
/// database, which on an exhaustively-matched 200-image project is ~20k
/// bincode payloads - long enough to stall a frame. It is therefore loaded on
/// a worker thread, the same way pipeline stages are run.
enum GraphState {
    Idle,
    Loading,
    Ready(Box<MatchGraph>),
    Failed(String),
}

struct App {
    project: PathBuf,
    scene: Option<Scene>,
    /// Kept alongside the `Scene` because the 2D overlay needs each image's
    /// raw keypoints and the outlier list needs point tracks, neither of which
    /// the render-oriented `Scene` carries.
    recon: Option<Reconstruction>,
    diag: Option<Diagnostics>,
    load_error: Option<String>,
    cam: OrbitCamera,
    selected: Option<usize>,
    selected_point: Option<usize>,
    show_points: bool,
    show_cameras: bool,
    show_plane: bool,
    point_size: i32,
    color_mode: ColorMode,
    /// Upper bound on a point's mean residual for it to be drawn, in pixels.
    residual_cutoff: f32,
    texture: Option<egui::TextureHandle>,
    run: RunState,
    expanded: BTreeSet<PathBuf>,
    show_log: bool,
    view: View,
    graph: Arc<Mutex<GraphState>>,
    loaded_image: Option<LoadedImage>,
    image_error: Option<String>,
    /// Residuals are sub-pixel on a good model and invisible at 1:1, so the
    /// overlay draws them multiplied by this.
    residual_exaggeration: f32,
    show_all_keypoints: bool,
    show_residual_vectors: bool,
    image_zoom: f32,
    image_pan: egui::Vec2,
    /// `[layout]` editor: one `LAYER_ROLES` index per path level. Sized from
    /// the source tree's actual depth the first time the section is drawn, so
    /// the user assigns roles to levels that exist rather than counting
    /// directories themselves.
    layout_layers: Vec<usize>,
    /// Name for the layout being added in the editor.
    layout_name: String,
    layout_default: bool,
    aruco: aruco::Tester,
    status: StageStatus,
    /// Stage options mirrored from the CLI's own defaults.
    detector: usize,
    camera_model: usize,
    pairing: usize,
    pipeline_kind: usize,
    export_format: usize,
}

const DETECTORS: [&str; 4] = ["sift", "aruco", "orb", "disk"];
const PAIRINGS: [&str; 3] = ["exhaustive", "sequential", "vocab-tree"];
const PIPELINES: [&str; 2] = ["incremental", "global"];
const EXPORTS: [&str; 2] = ["colmap-text", "nerf-transforms"];
/// The `project::Layer` roles as `sfm.toml` spells them.
const LAYER_ROLES: [&str; 4] = ["capture", "camera", "image", "ignore"];

/// Distinct hues for connected components in the match graph.
const COMPONENT_COLORS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(90, 160, 255),
    egui::Color32::from_rgb(240, 140, 60),
    egui::Color32::from_rgb(110, 200, 120),
    egui::Color32::from_rgb(220, 100, 190),
    egui::Color32::from_rgb(230, 200, 70),
    egui::Color32::from_rgb(120, 210, 220),
    egui::Color32::from_rgb(200, 90, 90),
    egui::Color32::from_rgb(160, 150, 230),
];

/// Which stages have already produced output, so the viewer can show where a
/// project stands instead of leaving the user to guess.
///
/// A beginner opening a half-finished project cannot tell from the buttons
/// alone whether features exist, whether matching ran, or why Map fails. These
/// are cheap filesystem and database facts, refreshed when a stage finishes
/// rather than polled every frame.
#[derive(Default, Clone, Copy)]
struct StageStatus {
    features: usize,
    pairs: usize,
    registered: usize,
}

impl StageStatus {
    fn read(project: &Path) -> Self {
        let Ok(p) = Project::open(project) else {
            return Self::default();
        };
        let mut st = StageStatus::default();
        if let Ok(db) = crate::db::Database::open(&p.database_path()) {
            st.features = db.list_images().map(|v| v.len()).unwrap_or(0);
            st.pairs = db.list_geometry_pairs().map(|v| v.len()).unwrap_or(0);
        }
        st.registered = sfm_io::read_colmap_model(&p.sparse_dir())
            .map(|r| r.images.len())
            .unwrap_or(0);
        st
    }
}

/// A pipeline-stage button that also says whether that stage has run.
///
/// The count is the point: "① Feature" alone cannot distinguish a project with
/// no features from one with 965 of them, and that is exactly the question
/// someone asks when a later stage fails.
fn stage_button(ui: &mut egui::Ui, label: &str, count: usize, noun: &str) -> egui::Response {
    let text = if count > 0 {
        egui::RichText::new(format!("{label}  ✔")).color(ok_color())
    } else {
        egui::RichText::new(label)
    };
    let r = ui.button(text);
    if count > 0 {
        r.on_hover_text(format!("{count} {noun} already"))
    } else {
        r.on_hover_text("not run yet for this project")
    }
}

/// One line telling the user what to do next, from what the project contains.
fn next_step_hint(st: &StageStatus) -> String {
    if st.features == 0 {
        "no features yet - start with ① Feature, or ▶ Run all".into()
    } else if st.pairs == 0 {
        format!("{} images have features - next is ② Match", st.features)
    } else if st.registered == 0 {
        format!("{} verified pairs - next is ③ Map", st.pairs)
    } else {
        format!(
            "{} images registered - inspect in the views below, or ④ Export",
            st.registered
        )
    }
}

fn warn_color() -> egui::Color32 {
    egui::Color32::from_rgb(235, 150, 40)
}

fn ok_color() -> egui::Color32 {
    egui::Color32::from_rgb(90, 190, 110)
}

impl App {
    fn new(project: PathBuf, view: View) -> Self {
        let mut app = App {
            project,
            scene: None,
            recon: None,
            diag: None,
            load_error: None,
            cam: OrbitCamera {
                target: nalgebra::Vector3::zeros(),
                distance: 5.0,
                yaw: 0.6,
                pitch: 0.35,
                fov_y: 50f64.to_radians(),
            },
            selected: None,
            selected_point: None,
            show_points: true,
            show_cameras: true,
            show_plane: false,
            point_size: 2,
            color_mode: ColorMode::Photometric,
            residual_cutoff: f32::MAX,
            texture: None,
            run: RunState::default(),
            expanded: BTreeSet::new(),
            show_log: true,
            view,
            graph: Arc::new(Mutex::new(GraphState::Idle)),
            loaded_image: None,
            image_error: None,
            residual_exaggeration: 10.0,
            show_all_keypoints: true,
            show_residual_vectors: true,
            image_zoom: 1.0,
            image_pan: egui::Vec2::ZERO,
            layout_layers: Vec::new(),
            layout_name: String::new(),
            layout_default: false,
            aruco: aruco::Tester::default(),
            status: StageStatus::default(),
            detector: 0,
            // Index of SIMPLE_RADIAL, matching the CLI default.
            camera_model: 2,
            pairing: 0,
            pipeline_kind: 0,
            export_format: 0,
        };
        app.reload();
        // Opening straight onto the image view with nothing selected would show
        // only a "select a camera" prompt, so pick the first one.
        if app.view == View::Image && app.selected.is_none() {
            if let Some(scene) = &app.scene {
                if !scene.images.is_empty() {
                    app.selected = Some(0);
                }
            }
        }
        app
    }

    /// Loads whatever `sfmtory map` last wrote, if anything.
    fn reload(&mut self) {
        self.status = StageStatus::read(&self.project);
        self.load_error = None;
        self.selected_point = None;
        self.loaded_image = None;
        self.image_error = None;
        if let Ok(mut g) = self.graph.lock() {
            *g = GraphState::Idle;
        }

        let project = match Project::open(&self.project) {
            Ok(p) => p,
            Err(e) => {
                self.load_error = Some(format!("{e}"));
                self.scene = None;
                self.recon = None;
                self.diag = None;
                return;
            }
        };
        let dir = project.sparse_dir();
        if !dir.join("cameras.txt").exists() {
            self.scene = None;
            self.recon = None;
            self.diag = None;
            self.load_error = Some(format!(
                "No reconstruction yet at {}.\nRun the stages above, or open a project that has one.",
                dir.display()
            ));
            return;
        }
        match sfm_io::read_colmap_model(&dir) {
            Ok(recon) => {
                let s = Scene::from_reconstruction(&recon);
                self.cam = OrbitCamera::framing(&s);
                self.selected = None;
                let diag = Diagnostics::compute(
                    &recon,
                    &initial_cameras(&project),
                    &pinned_cameras(&project, &recon),
                );
                // Start with nothing filtered out, whatever this model's
                // residual range turns out to be.
                self.residual_cutoff = diag.residuals.max.max(1e-6) as f32;
                self.show_plane = diag
                    .plane
                    .as_ref()
                    .is_some_and(|p| p.verdict != CoverageVerdict::NotPlanar);
                self.diag = Some(diag);
                self.scene = Some(s);
                self.recon = Some(recon);
            }
            Err(e) => {
                self.scene = None;
                self.recon = None;
                self.diag = None;
                self.load_error = Some(format!("failed to read {}: {e}", dir.display()));
            }
        }
    }

    /// Kicks off the match-graph read if it has not been started yet.
    fn ensure_graph(&self, ctx: &egui::Context) {
        match self.graph.lock() {
            Ok(mut g) if matches!(*g, GraphState::Idle) => *g = GraphState::Loading,
            _ => return,
        }
        let slot = self.graph.clone();
        let ctx = ctx.clone();
        let project = self.project.clone();
        // The graph is drawn against the *database*, not the reconstruction, so
        // it is available after `match` and before `map` has ever succeeded -
        // which is exactly when a disconnected graph is worth seeing.
        let registered: BTreeSet<u32> = self
            .recon
            .as_ref()
            .map(|r| r.images.keys().copied().collect())
            .unwrap_or_default();
        std::thread::spawn(move || {
            let result = Project::open(&project)
                .map_err(|e| e.to_string())
                .and_then(|p| {
                    let db = p.database_path();
                    if !db.exists() {
                        return Err(format!(
                            "no project database at {} - run `feature` and `match` first",
                            db.display()
                        ));
                    }
                    MatchGraph::load(&db, &registered).map_err(|e| e.to_string())
                });
            if let Ok(mut g) = slot.lock() {
                *g = match result {
                    Ok(graph) => GraphState::Ready(Box::new(graph)),
                    Err(e) => GraphState::Failed(e),
                };
            }
            ctx.request_repaint();
        });
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("stages").show(ctx, |ui| {
            let busy = self.run.is_running();
            let st = self.status;
            ui.add_space(4.0);

            // Row one is the pipeline in order, each stage showing whether it
            // has already produced something. A project half-finished used to
            // look exactly like a project not started.
            ui.horizontal_wrapped(|ui| {
                ui.strong("Pipeline");
                ui.add_enabled_ui(!busy, |ui| {
                    if ui
                        .button("▶ Run all")
                        .on_hover_text("feature → match → map → export, with the options set here")
                        .clicked()
                    {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "run".into(),
                                "--detector".into(),
                                DETECTORS[self.detector].into(),
                                "--camera-model".into(),
                                crate::CAMERA_MODELS[self.camera_model].into(),
                                "--pairing".into(),
                                PAIRINGS[self.pairing].into(),
                                "--pipeline".into(),
                                PIPELINES[self.pipeline_kind].into(),
                            ],
                        );
                    }
                    ui.separator();

                    egui::ComboBox::from_id_salt("detector")
                        .selected_text(DETECTORS[self.detector])
                        .width(70.0)
                        .show_ui(ui, |ui| {
                            for (i, d) in DETECTORS.iter().enumerate() {
                                ui.selectable_value(&mut self.detector, i, *d);
                            }
                        });
                    egui::ComboBox::from_id_salt("cameramodel")
                        .selected_text(crate::CAMERA_MODELS[self.camera_model])
                        .width(130.0)
                        .show_ui(ui, |ui| {
                            for (i, m) in crate::CAMERA_MODELS.iter().enumerate() {
                                ui.selectable_value(&mut self.camera_model, i, *m);
                            }
                        })
                        .response
                        .on_hover_text(
                            "Camera model for images no [[cameras]] entry covers.                              SIMPLE_RADIAL suits a narrow lens; a wide one needs OPENCV, and                              past ~100 degrees diagonal, OPENCV_FISHEYE.",
                        );
                    if stage_button(ui, "① Feature", st.features, "images").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "feature".into(),
                                "--detector".into(),
                                DETECTORS[self.detector].into(),
                                "--camera-model".into(),
                                crate::CAMERA_MODELS[self.camera_model].into(),
                            ],
                        );
                    }

                    egui::ComboBox::from_id_salt("pairing")
                        .selected_text(PAIRINGS[self.pairing])
                        .width(120.0)
                        .show_ui(ui, |ui| {
                            for (i, p) in PAIRINGS.iter().enumerate() {
                                ui.selectable_value(&mut self.pairing, i, *p);
                            }
                        });
                    if stage_button(ui, "② Match", st.pairs, "pairs").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "match".into(),
                                "--pairing".into(),
                                PAIRINGS[self.pairing].into(),
                            ],
                        );
                    }

                    egui::ComboBox::from_id_salt("pipeline")
                        .selected_text(PIPELINES[self.pipeline_kind])
                        .width(95.0)
                        .show_ui(ui, |ui| {
                            for (i, p) in PIPELINES.iter().enumerate() {
                                ui.selectable_value(&mut self.pipeline_kind, i, *p);
                            }
                        });
                    if stage_button(ui, "③ Map", st.registered, "registered").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "map".into(),
                                "--pipeline".into(),
                                PIPELINES[self.pipeline_kind].into(),
                            ],
                        );
                    }

                    egui::ComboBox::from_id_salt("export")
                        .selected_text(EXPORTS[self.export_format])
                        .width(120.0)
                        .show_ui(ui, |ui| {
                            for (i, f) in EXPORTS.iter().enumerate() {
                                ui.selectable_value(&mut self.export_format, i, *f);
                            }
                        });
                    if ui.button("④ Export").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "export".into(),
                                "--format".into(),
                                EXPORTS[self.export_format].into(),
                            ],
                        );
                    }
                });
                if busy {
                    ui.spinner();
                    let stage = self
                        .run
                        .last_stage
                        .lock()
                        .map(|s| s.clone())
                        .unwrap_or_default();
                    ui.label(format!("running {stage}..."));
                }
            });

            // Row two is everything that inspects or calibrates rather than
            // builds, kept off the main path so the four numbered steps read
            // as the thing to do first.
            ui.horizontal_wrapped(|ui| {
                ui.strong("Analysis");
                ui.add_enabled_ui(!busy, |ui| {
                    if ui
                        .button("Init-cam")
                        .on_hover_text("estimate a focal length before reconstructing")
                        .clicked()
                    {
                        self.run.spawn(&self.project, vec!["init-cam".into()]);
                    }
                    if ui
                        .button("Rig")
                        .on_hover_text(
                            "for fixed cameras and a target that moved between captures:                              reconstruct each capture, align them, and report how well the                              cameras agree",
                        )
                        .clicked()
                    {
                        self.run.spawn(&self.project, vec!["rig".into()]);
                    }
                    if ui
                        .button("Select-model")
                        .on_hover_text(
                            "score every camera model on held-out reprojection error and                              report which one this lens actually needs",
                        )
                        .clicked()
                    {
                        self.run.spawn(&self.project, vec!["select-model".into()]);
                    }
                    if ui
                        .button("Eval")
                        .on_hover_text("reprojection statistics for the current model")
                        .clicked()
                    {
                        self.run.spawn(&self.project, vec!["eval".into()]);
                    }
                });
                ui.separator();
                if ui.button("⟳ Reload").clicked() {
                    self.reload();
                }
                // The one-line answer to "what should I do next?".
                ui.label(
                    egui::RichText::new(next_step_hint(&st))
                        .small()
                        .color(if st.registered > 0 { ok_color() } else { warn_color() }),
                );
            });
            ui.add_space(4.0);
        });
    }

    fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("explorer")
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Project");
                ui.label(
                    egui::RichText::new(self.project.display().to_string())
                        .small()
                        .weak(),
                );
                if ui.button("Open parent directory").clicked() {
                    if let Some(p) = self.project.parent().map(Path::to_path_buf) {
                        self.project = p;
                        self.reload();
                    }
                }
                ui.separator();
                self.layout_section(ui);
                ui.separator();
                egui::ScrollArea::vertical().id_salt("tree").show(ui, |ui| {
                    let root = self.project.clone();
                    self.dir_tree(ui, &root, 0);
                });
            });
    }

    /// Declare, select, build and delete dataset layouts without leaving the
    /// viewer.
    ///
    /// Every button here runs the same `sfmtory dataset ...` subcommand you
    /// would type, as a child process, so the file-editing rules - migrating a
    /// lone `[layout]`, preserving comments, refusing a duplicate name - live
    /// in one place and the viewer cannot drift from them.
    fn layout_section(&mut self, ui: &mut egui::Ui) {
        let project = Project::open(&self.project).ok();
        let layouts = project
            .as_ref()
            .map(|p| p.all_layouts())
            .unwrap_or_default();
        let linked = project
            .as_ref()
            .map(|p| p.linked_images_dir().is_dir())
            .unwrap_or(false);
        let busy = self.run.is_running();
        let mut action: Option<Vec<String>> = None;

        egui::CollapsingHeader::new("Dataset layout")
            .default_open(!layouts.is_empty() && !linked)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "For a dataset whose folder shape cannot be inferred - one folder \
                         per capture with a file per camera, a folder per camera holding a \
                         video's frames, or ids encoded in the names.",
                    )
                    .small()
                    .weak(),
                );
                ui.add_space(4.0);

                if layouts.is_empty() {
                    ui.label("no layouts declared - the usual case");
                } else {
                    for l in &layouts {
                        ui.horizontal(|ui| {
                            let tag = if l.default { " (default)" } else { "" };
                            ui.label(egui::RichText::new(format!("{}{tag}", l.name)).strong());
                            if ui
                                .add_enabled(!busy, egui::Button::new("link").small())
                                .on_hover_text("build this layout's symlink tree")
                                .clicked()
                            {
                                action = Some(vec![
                                    "dataset".into(),
                                    "link".into(),
                                    "--force".into(),
                                    "--layout".into(),
                                    l.name.clone(),
                                ]);
                            }
                            if ui
                                .add_enabled(!busy, egui::Button::new("remove").small())
                                .clicked()
                            {
                                action = Some(vec![
                                    "dataset".into(),
                                    "remove-layout".into(),
                                    l.name.clone(),
                                ]);
                            }
                        });
                        ui.label(
                            egui::RichText::new(format!("   [{}]", l.layout.layers.join(", ")))
                                .small()
                                .weak(),
                        );
                    }
                    ui.label(
                        egui::RichText::new(if linked {
                            "link tree built"
                        } else {
                            "link tree not built - press link"
                        })
                        .small()
                        .color(if linked {
                            ok_color()
                        } else {
                            warn_color()
                        }),
                    );
                }

                ui.add_space(6.0);
                ui.separator();
                ui.strong("Add a layout");
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.layout_name)
                            .desired_width(110.0)
                            .hint_text("rig"),
                    );
                });
                // One row per path level, sized from the tree itself: the depth
                // is a fact about the dataset rather than a choice.
                if self.layout_layers.is_empty() {
                    let depth = project
                        .as_ref()
                        .and_then(|p| crate::layout::probe_depth(&p.config.images_dir))
                        .unwrap_or(2);
                    self.layout_layers = (0..depth)
                        .map(|i| {
                            if i == 0 {
                                0
                            } else if i + 1 == depth {
                                1
                            } else {
                                3
                            }
                        })
                        .collect();
                }
                for i in 0..self.layout_layers.len() {
                    ui.horizontal(|ui| {
                        let last = i + 1 == self.layout_layers.len();
                        ui.label(format!("level {}", i + 1));
                        ui.label(
                            egui::RichText::new(if last { "(file)" } else { "(dir) " })
                                .small()
                                .weak(),
                        );
                        egui::ComboBox::from_id_salt(("layer", i))
                            .selected_text(LAYER_ROLES[self.layout_layers[i]])
                            .width(90.0)
                            .show_ui(ui, |ui| {
                                for (k, name) in LAYER_ROLES.iter().enumerate() {
                                    ui.selectable_value(&mut self.layout_layers[i], k, *name);
                                }
                            });
                    });
                }
                ui.horizontal(|ui| {
                    if ui.small_button("+ level").clicked() {
                        self.layout_layers.push(3);
                    }
                    if ui.small_button("- level").clicked() && self.layout_layers.len() > 1 {
                        self.layout_layers.pop();
                    }
                    ui.checkbox(&mut self.layout_default, "default");
                });
                if ui
                    .add_enabled(
                        !busy && !self.layout_name.trim().is_empty(),
                        egui::Button::new("Add layout"),
                    )
                    .clicked()
                {
                    let mut args = vec![
                        "dataset".into(),
                        "add-layout".into(),
                        "--name".into(),
                        self.layout_name.trim().to_string(),
                        "--layers".into(),
                    ];
                    args.extend(
                        self.layout_layers
                            .iter()
                            .map(|&i| LAYER_ROLES[i].to_string()),
                    );
                    if self.layout_default {
                        args.push("--default".into());
                    }
                    action = Some(args);
                }
                ui.label(
                    egui::RichText::new(
                        "A level can also be a pattern like cam{camera}_{image} when one \
                         name carries two ids; add those by editing sfm.toml.",
                    )
                    .small()
                    .weak(),
                );
            });

        if let Some(args) = action {
            self.run.spawn(&self.project, args);
        }
    }

    fn dir_tree(&mut self, ui: &mut egui::Ui, dir: &Path, depth: usize) {
        // Guard against a deep or symlinked tree turning browsing into a hang.
        if depth > 6 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort_by_key(|p| (!p.is_dir(), p.file_name().map(|s| s.to_os_string())));
        for path in entries {
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if path.is_dir() {
                let open = self.expanded.contains(&path);
                let label = format!("{} {}", if open { "▾" } else { "▸" }, name);
                let resp = ui.selectable_label(false, label);
                if resp.clicked() {
                    if open {
                        self.expanded.remove(&path);
                    } else {
                        self.expanded.insert(path.clone());
                    }
                }
                // Double-clicking a directory makes it the working project,
                // which is the quickest way to move between datasets.
                if resp.double_clicked() {
                    self.project = path.clone();
                    self.reload();
                }
                if self.expanded.contains(&path) {
                    ui.indent(&path, |ui| self.dir_tree(ui, &path, depth + 1));
                }
            } else {
                ui.label(egui::RichText::new(format!("   {name}")).weak());
            }
        }
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("properties")
            .default_width(360.0)
            .show(ctx, |ui| {
                let Some(scene) = &self.scene else {
                    ui.heading("Properties");
                    ui.separator();
                    ui.label("No model loaded.");
                    return;
                };
                ui.horizontal(|ui| {
                    ui.label(format!("{} images", scene.images.len()));
                    ui.separator();
                    ui.label(format!("{} points", scene.points.len()));
                    ui.separator();
                    ui.label(format!("{} camera(s)", scene.cameras.len()));
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("props")
                    .show(ui, |ui| {
                        self.selection_section(ui);
                        ui.add_space(4.0);
                        self.calibration_section(ui);
                        ui.add_space(4.0);
                        self.diversity_section(ui);
                        ui.add_space(4.0);
                        self.outlier_section(ui);
                    });
            });
    }

    // ----- right-panel sections -------------------------------------------

    fn selection_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Selection")
            .default_open(true)
            .show(ui, |ui| {
                if self.selected_point.is_some() {
                    self.point_details(ui);
                    ui.add_space(4.0);
                    ui.separator();
                }
                let Some(scene) = &self.scene else { return };
                let Some(sel) = self.selected else {
                    ui.label("Select a camera in the 3D view, or from the list below.");
                    ui.add_space(6.0);
                    let mut pick = None;
                    egui::ScrollArea::vertical()
                        .id_salt("imglist")
                        .max_height(260.0)
                        .show(ui, |ui| {
                            for (i, im) in scene.images.iter().enumerate() {
                                if ui.selectable_label(false, &im.name).clicked() {
                                    pick = Some(i);
                                }
                            }
                        });
                    if let Some(i) = pick {
                        self.select_image(i);
                    }
                    return;
                };
                let im = &scene.images[sel];
                ui.strong(&im.name);
                ui.label(format!("image id {}  ·  camera id {}", im.id, im.camera_id));
                if let Some(d) = self.diag.as_ref().and_then(|d| d.images.get(&im.id)) {
                    ui.label(format!(
                        "{} of {} keypoints triangulated",
                        d.num_observations, d.num_keypoints
                    ));
                    ui.label(format!(
                        "residual  mean {:.3}px   max {:.3}px",
                        d.mean_residual_px, d.max_residual_px
                    ));
                } else {
                    ui.label(format!("{} observations", im.num_observations));
                }
                ui.add_space(6.0);

                ui.strong("Intrinsics");
                if let Some(cam) = scene.cameras.get(&im.camera_id) {
                    ui.label(format!("model: {}", cam.model.name()));
                    ui.label(format!("size: {} x {}", cam.width, cam.height));
                    let (fx, fy) = cam.model.focal_lengths();
                    let (cx, cy) = cam.model.principal_point();
                    ui.label(format!("fx {fx:.3}   fy {fy:.3}"));
                    ui.label(format!("cx {cx:.3}   cy {cy:.3}"));
                    let d = cam.model.opencv_distortion();
                    if d.iter().any(|v| *v != 0.0) {
                        ui.label(format!(
                            "distortion: [{}]",
                            d.iter()
                                .map(|v| format!("{v:.5}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                } else {
                    ui.label("(camera not found in model)");
                }

                ui.add_space(6.0);
                ui.strong("Extrinsics (world-to-camera)");
                let q = im.quaternion;
                ui.label(format!(
                    "q  w {:.6}  x {:.6}\n   y {:.6}  z {:.6}",
                    q[0], q[1], q[2], q[3]
                ));
                let t = im.translation;
                ui.label(format!("t  [{:.6}, {:.6}, {:.6}]", t[0], t[1], t[2]));
                ui.label(format!(
                    "centre  [{:.4}, {:.4}, {:.4}]",
                    im.center.x, im.center.y, im.center.z
                ));

                ui.add_space(8.0);
                ui.separator();
                ui.strong(format!("Other images on camera {}", im.camera_id));
                let cam_id = im.camera_id;
                let mut pick = None;
                egui::ScrollArea::vertical()
                    .id_salt("siblings")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for (i, other) in scene.images.iter().enumerate() {
                            if other.camera_id != cam_id {
                                continue;
                            }
                            if ui.selectable_label(i == sel, &other.name).clicked() {
                                pick = Some(i);
                            }
                        }
                    });
                if let Some(i) = pick {
                    self.select_image(i);
                }
            });
    }

    /// The images observing the selected 3D point, with each observation's own
    /// residual - the "click a point to see who sees it" half of outlier
    /// inspection, and the fastest way to tell a badly-triangulated point from
    /// one bad correspondence dragging a good point.
    fn point_details(&mut self, ui: &mut egui::Ui) {
        let mut rows: Vec<(String, f64, usize)> = Vec::new();
        let mut header: Option<(u64, [f32; 3], usize, f64)> = None;
        if let (Some(scene), Some(diag), Some(recon), Some(pi)) =
            (&self.scene, &self.diag, &self.recon, self.selected_point)
        {
            if let Some(&pid) = scene.point_ids.get(pi) {
                let xyz = scene.points[pi].0;
                header = Some((
                    pid,
                    [xyz.x, xyz.y, xyz.z],
                    recon.points3d.get(&pid).map(|p| p.track.len()).unwrap_or(0),
                    diag.residuals.point_mean.get(&pid).copied().unwrap_or(0.0),
                ));
                if let Some(idx) = diag.residuals.by_point.get(&pid) {
                    for &i in idx {
                        let o = &diag.residuals.all[i];
                        let name = recon
                            .images
                            .get(&o.image_id)
                            .map(|im| im.name.clone())
                            .unwrap_or_else(|| format!("image {}", o.image_id));
                        let scene_index = scene
                            .images
                            .iter()
                            .position(|s| s.id == o.image_id)
                            .unwrap_or(usize::MAX);
                        rows.push((name, o.residual_px, scene_index));
                    }
                }
            }
        }
        let Some((pid, xyz, track_len, mean)) = header else {
            return;
        };
        let max = self.diag.as_ref().map(|d| d.residuals.max).unwrap_or(1.0);

        ui.horizontal(|ui| {
            ui.strong(format!("Point {pid}"));
            if ui.small_button("clear").clicked() {
                self.selected_point = None;
            }
        });
        ui.label(format!(
            "xyz  [{:.4}, {:.4}, {:.4}]",
            xyz[0], xyz[1], xyz[2]
        ));
        ui.label(format!(
            "track length {track_len}   ·   mean residual {mean:.3}px"
        ));
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ui.add_space(2.0);
        ui.label(egui::RichText::new("observed in").small().weak());
        let mut pick = None;
        for (name, residual, scene_index) in rows {
            let c = scene::error_color(residual, max);
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 1.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                if ui
                    .selectable_label(false, format!("{name}   {residual:.3}px"))
                    .clicked()
                    && scene_index != usize::MAX
                {
                    pick = Some(scene_index);
                }
            });
        }
        if let Some(i) = pick {
            self.select_image(i);
            self.view = View::Image;
        }
    }

    fn calibration_section(&mut self, ui: &mut egui::Ui) {
        if self.diag.is_none() {
            return;
        }
        egui::CollapsingHeader::new("Calibration quality")
            .default_open(true)
            .show(ui, |ui| {
                let diag = self.diag.as_ref().unwrap();
                for cam in &diag.cameras {
                    let text = egui::RichText::new(cam.verdict.headline()).strong();
                    ui.label(if cam.verdict.is_warning() {
                        text.color(warn_color())
                    } else {
                        text
                    });
                    if let Some(evidence) = cam.evidence() {
                        ui.label(egui::RichText::new(evidence).small().weak());
                    }
                    ui.label(format!(
                        "camera {}  ({})  ·  {} image(s)  ·  {} observation(s)",
                        cam.camera_id, cam.model_name, cam.num_images, cam.num_observations
                    ));
                    // Field of view decides whether the chosen model can even
                    // represent this lens, and it is not otherwise anywhere in
                    // the UI - the focal alone does not say, because it means
                    // nothing without the sensor size beside it.
                    if let Some((hfov, dfov)) = cam.field_of_view_deg {
                        let advice = if dfov > 100.0 {
                            Some("wide enough that OPENCV_FISHEYE is worth testing")
                        } else if dfov > 70.0 && cam.model_name.starts_with("SIMPLE_") {
                            Some("too wide for a single radial term - try OPENCV")
                        } else {
                            None
                        };
                        let text =
                            format!("field of view  {hfov:.0}° horizontal, {dfov:.0}° diagonal");
                        match advice {
                            Some(a) => {
                                ui.label(egui::RichText::new(text).color(warn_color()));
                                ui.label(egui::RichText::new(format!("   {a}")).small().weak());
                            }
                            None => {
                                ui.label(text);
                            }
                        }
                    }
                    match cam.focal_initial {
                        Some((ix, iy)) => ui.label(format!(
                            "focal  initial {:.2} / {:.2}   →   final {:.2} / {:.2}",
                            ix, iy, cam.focal_final.0, cam.focal_final.1
                        )),
                        None => ui.label(format!(
                            "focal  final {:.2} / {:.2}   (no initial guess on record)",
                            cam.focal_final.0, cam.focal_final.1
                        )),
                    };
                    ui.add_space(6.0);
                }

                ui.separator();
                ui.strong("Track lengths");
                let t = &diag.tracks;
                if t.total_points == 0 {
                    ui.label("no triangulated points");
                    return;
                }
                ui.label(format!(
                    "{} points  ·  min {}  median {:.0}  mean {:.2}  max {}",
                    t.total_points, t.min, t.median, t.mean, t.max
                ));
                if t.max <= 2 {
                    ui.label(
                        egui::RichText::new(
                            "every track has at most 2 observations - no 3D point is seen by \
                             enough views to observe a shared focal length",
                        )
                        .small()
                        .color(warn_color()),
                    );
                }
                histogram(ui, t, 90.0);

                ui.add_space(6.0);
                ui.strong("Observations per image");
                let recon = self.recon.as_ref();
                let mut rows: Vec<_> = diag.images.values().collect();
                rows.sort_by_key(|d| d.num_observations);
                let max = rows.last().map(|d| d.num_observations).unwrap_or(1).max(1) as f32;
                egui::ScrollArea::vertical()
                    .id_salt("obscounts")
                    .max_height(150.0)
                    .show(ui, |ui| {
                        for d in rows {
                            let name = recon
                                .and_then(|r| r.images.get(&d.image_id))
                                .map(|im| im.name.as_str())
                                .unwrap_or("?");
                            ui.horizontal(|ui| {
                                bar(
                                    ui,
                                    d.num_observations as f32 / max,
                                    70.0,
                                    egui::Color32::from_rgb(90, 140, 220),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{:>5}/{:<5} {name}",
                                        d.num_observations, d.num_keypoints
                                    ))
                                    .monospace()
                                    .small(),
                                );
                            });
                        }
                    });
            });
    }

    fn diversity_section(&mut self, ui: &mut egui::Ui) {
        if self.diag.as_ref().and_then(|d| d.plane.as_ref()).is_none() {
            return;
        }
        let default_open = self
            .diag
            .as_ref()
            .and_then(|d| d.plane.as_ref())
            .is_some_and(|p| p.verdict != CoverageVerdict::NotPlanar);
        let mut go_to_coverage = false;
        egui::CollapsingHeader::new("Viewing-angle diversity")
            .default_open(default_open)
            .show(ui, |ui| {
                let plane = self.diag.as_ref().unwrap().plane.as_ref().unwrap();
                ui.label(format!(
                    "best-fit plane flatness {:.4}   (planar below {:.2})",
                    plane.flatness,
                    diagnostics::PLANAR_FLATNESS
                ));
                ui.label(
                    egui::RichText::new(format!(
                        "normal  [{:.4}, {:.4}, {:.4}]",
                        plane.normal.x, plane.normal.y, plane.normal.z
                    ))
                    .small()
                    .weak(),
                );
                match &plane.verdict {
                    CoverageVerdict::NotPlanar => {
                        ui.label(
                            "the reconstruction is not planar, so single-plane (Zhang-style) \
                             calibration diversity does not apply here",
                        );
                    }
                    CoverageVerdict::Adequate => {
                        ui.label(
                            egui::RichText::new("angular coverage looks adequate")
                                .color(ok_color())
                                .strong(),
                        );
                    }
                    CoverageVerdict::Narrow(msg) => {
                        ui.label(
                            egui::RichText::new("narrow angular coverage")
                                .color(warn_color())
                                .strong(),
                        );
                        ui.label(egui::RichText::new(msg).small());
                    }
                }
                ui.label(format!(
                    "tilt to plane  {:.1}° … {:.1}°   ·   {} of 8 azimuth sectors",
                    plane.tilt_min_deg, plane.tilt_max_deg, plane.azimuth_sectors_covered
                ));
                // The axis spread is the criterion the verdict actually turns
                // on, so it is stated next to the numbers that do not.
                ui.label(format!(
                    "tilt-axis spread  {:.1}°   (the quantity Zhang's method is                      degenerate in)",
                    plane.axis_spread_deg
                ));
                tilt_strip(ui, plane);
                if ui.button("Show coverage plot").clicked() {
                    go_to_coverage = true;
                }
            });
        if go_to_coverage {
            self.view = View::Coverage;
        }
    }

    fn outlier_section(&mut self, ui: &mut egui::Ui) {
        if self.diag.is_none() {
            return;
        }
        let mut pick: Option<(u64, u32)> = None;
        egui::CollapsingHeader::new("Reprojection & outliers")
            .default_open(false)
            .show(ui, |ui| {
                let diag = self.diag.as_ref().unwrap();
                let r = &diag.residuals;
                ui.label(
                    egui::RichText::new(format!(
                        "mean {:.3}px   median {:.3}px   p95 {:.3}px   max {:.3}px",
                        r.mean, r.median, r.p95, r.max
                    ))
                    .monospace()
                    .small(),
                );
                ui.label(format!("{} observations", r.all.len()));
                if r.num_behind_camera > 0 {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} observation(s) fall behind their camera and are excluded",
                            r.num_behind_camera
                        ))
                        .small()
                        .color(warn_color()),
                    );
                }
                // A mean says nothing about whether a model is uniformly decent
                // or mostly excellent with a few broken points, so say which.
                if r.p95 > 0.0 && r.max > r.p95 * 5.0 {
                    ui.label(
                        egui::RichText::new(format!(
                            "the worst observation is {:.0}x the 95th percentile: this model is \
                             mostly excellent with a few broken points, not uniformly {:.2}px",
                            r.max / r.p95,
                            r.mean
                        ))
                        .small()
                        .color(warn_color()),
                    );
                }
                ui.add_space(4.0);
                ui.strong("Worst observations");
                let mut worst: Vec<usize> = (0..r.all.len()).collect();
                worst.sort_by(|&a, &b| {
                    r.all[b]
                        .residual_px
                        .partial_cmp(&r.all[a].residual_px)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                worst.truncate(40);
                let recon = self.recon.as_ref();
                egui::ScrollArea::vertical()
                    .id_salt("worst")
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for i in worst {
                            let o = &r.all[i];
                            let name = recon
                                .and_then(|rc| rc.images.get(&o.image_id))
                                .map(|im| im.name.as_str())
                                .unwrap_or("?");
                            if ui
                                .selectable_label(
                                    false,
                                    egui::RichText::new(format!(
                                        "{:>9.2}px  pt {:<7} {name}",
                                        o.residual_px, o.point_id
                                    ))
                                    .monospace()
                                    .small(),
                                )
                                .clicked()
                            {
                                pick = Some((o.point_id, o.image_id));
                            }
                        }
                    });
            });
        if let Some((point_id, image_id)) = pick {
            let found = self.scene.as_ref().map(|scene| {
                (
                    scene.point_ids.iter().position(|&id| id == point_id),
                    scene.images.iter().position(|im| im.id == image_id),
                )
            });
            if let Some((p, i)) = found {
                self.selected_point = p;
                if let Some(i) = i {
                    self.select_image(i);
                }
            }
        }
    }

    // ----- central views ---------------------------------------------------

    fn select_image(&mut self, index: usize) {
        if self.selected != Some(index) {
            self.selected = Some(index);
            // Drop the cached texture so the next Image frame decodes the new
            // selection rather than showing the previous photograph.
            self.loaded_image = None;
            self.image_error = None;
            self.image_zoom = 1.0;
            self.image_pan = egui::Vec2::ZERO;
        }
    }

    fn central(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view, View::Start, "Start here");
                ui.selectable_value(&mut self.view, View::Scene3d, "3D");
                ui.selectable_value(&mut self.view, View::Image, "Image");
                ui.selectable_value(&mut self.view, View::Graph, "Match graph");
                ui.selectable_value(&mut self.view, View::Coverage, "Coverage");
                ui.selectable_value(&mut self.view, View::Aruco, "ArUco test");
                ui.separator();
                ui.checkbox(&mut self.show_log, "Log");
            });
            ui.separator();

            let avail = ui.available_size();
            let h = if self.show_log {
                (avail.y - 160.0).max(120.0)
            } else {
                avail.y
            };

            match self.view {
                View::Start => self.view_start(ui, h),
                View::Scene3d => self.view_3d(ctx, ui, h),
                View::Image => self.view_image(ctx, ui, h),
                View::Graph => self.view_graph(ctx, ui, h),
                View::Coverage => self.view_coverage(ui, h),
                View::Aruco => self.view_aruco(ctx, ui, h),
            }

            if self.show_log {
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("log")
                    .stick_to_bottom(true)
                    .max_height(140.0)
                    .show(ui, |ui| {
                        if let Ok(lines) = self.run.log.lock() {
                            for line in lines.iter() {
                                ui.label(egui::RichText::new(line).monospace().small());
                            }
                        }
                    });
            }
        });
    }

    /// The first screen: the next command, then whether its output will be a
    /// calibration or a picture of one. See `gui::start` for why this is the
    /// default view rather than the 3D scene.
    fn view_start(&mut self, ui: &mut egui::Ui, height: f32) {
        let st = StageStatus::read(&self.project);
        let readiness = start::Readiness::read(&self.project);
        let busy = self.run.is_running();

        egui::ScrollArea::vertical()
            .id_salt("start")
            .max_height(height)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.heading("Calibrate a camera rig");
                ui.label(
                    egui::RichText::new(format!(
                        "{} image(s), {} capture(s), {} camera(s) in this project",
                        readiness.num_images, readiness.num_captures, readiness.num_cameras
                    ))
                    .weak(),
                );
                ui.add_space(10.0);

                // --- the four stages, in order, with the next one highlighted
                ui.strong("Pipeline");
                ui.add_space(4.0);
                let steps: [(&str, bool, Vec<String>, &str); 4] = [
                    (
                        "① Feature - find markers or keypoints in every image",
                        st.features > 0,
                        vec!["feature".into(), "--detector".into(), "aruco".into()],
                        "Writes one feature set per image into the project database.",
                    ),
                    (
                        "② Match - find and verify which images share what",
                        st.pairs > 0,
                        vec!["match".into(), "--pairing".into(), "exhaustive".into()],
                        "Keeps only pairs that survive a geometric consistency check.",
                    ),
                    (
                        "③ Map - solve poses, points and intrinsics",
                        st.registered > 0,
                        vec!["map".into(), "--pipeline".into(), "incremental".into()],
                        "This is the step that produces the calibration.",
                    ),
                    (
                        "④ Export - write the model out",
                        false,
                        vec!["export".into()],
                        "COLMAP text, PLY, or NeRF-style transforms.",
                    ),
                ];
                let next = steps.iter().position(|(_, done, ..)| !*done);
                for (i, (label, done, cmd, why)) in steps.iter().enumerate() {
                    let is_next = next == Some(i);
                    ui.horizontal(|ui| {
                        let text = if *done {
                            egui::RichText::new(format!("{label}  ✔")).color(ok_color())
                        } else if is_next {
                            egui::RichText::new(*label).strong()
                        } else {
                            egui::RichText::new(*label).weak()
                        };
                        ui.label(text);
                        if ui
                            .add_enabled(!busy, egui::Button::new("Run"))
                            .on_hover_text(cmd.join(" "))
                            .clicked()
                        {
                            self.run.spawn(&self.project, cmd.clone());
                        }
                    });
                    ui.label(egui::RichText::new(format!("      {why}")).weak().small());
                }
                ui.add_space(6.0);
                if ui
                    .add_enabled(!busy, egui::Button::new("▶ Run all"))
                    .on_hover_text("feature -> match -> map -> export")
                    .clicked()
                {
                    self.run.spawn(
                        &self.project,
                        vec![
                            "run".into(),
                            "--detector".into(),
                            DETECTORS[self.detector].into(),
                            "--pairing".into(),
                            PAIRINGS[self.pairing].into(),
                            "--pipeline".into(),
                            PIPELINES[self.pipeline_kind].into(),
                        ],
                    );
                }

                ui.add_space(14.0);
                ui.separator();
                ui.add_space(6.0);

                // --- the part that decides whether the numbers mean anything
                ui.strong("Will the result be a calibration?");
                ui.label(
                    egui::RichText::new(
                        "A reconstruction that never refined its intrinsics still renders a tidy \
                         point cloud and confident-looking cameras. These checks are the \
                         difference.",
                    )
                    .weak()
                    .small(),
                );
                ui.add_space(6.0);

                if readiness.checks.is_empty() {
                    ui.label(egui::RichText::new("Nothing to check yet - no images found.").weak());
                }
                for check in &readiness.checks {
                    ui.horizontal(|ui| {
                        if check.ok {
                            ui.label(egui::RichText::new("✔").color(ok_color()));
                        } else {
                            ui.label(egui::RichText::new("!").color(warn_color()).strong());
                        }
                        ui.label(egui::RichText::new(check.what).strong());
                        ui.label(egui::RichText::new(&check.state).weak());
                    });
                    if let Some(fix) = &check.fix {
                        ui.label(egui::RichText::new(format!("      {fix}")).small());
                    }
                    if let Some(cmd) = &check.command {
                        ui.horizontal(|ui| {
                            ui.add_space(20.0);
                            if ui
                                .add_enabled(
                                    !busy,
                                    egui::Button::new(format!("Run: {}", cmd.join(" "))),
                                )
                                .clicked()
                            {
                                self.run.spawn(&self.project, cmd.clone());
                            }
                        });
                    }
                    ui.add_space(4.0);
                }

                if readiness.all_ok() && st.registered > 0 {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "All checks pass. Inspect the result in the 3D, Image and Match \
                             graph views.",
                        )
                        .color(ok_color()),
                    );
                }
            });
    }

    fn view_3d(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, height: f32) {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_points, "3D points");
            ui.checkbox(&mut self.show_cameras, "Cameras");
            ui.separator();
            ui.label("point size");
            ui.add(egui::Slider::new(&mut self.point_size, 1..=5).show_value(false));
            ui.separator();
            ui.label("colour");
            egui::ComboBox::from_id_salt("colormode")
                .selected_text(match self.color_mode {
                    ColorMode::Photometric => "photometric",
                    ColorMode::Residual => "reprojection error",
                })
                .width(150.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.color_mode,
                        ColorMode::Photometric,
                        "photometric",
                    );
                    ui.selectable_value(
                        &mut self.color_mode,
                        ColorMode::Residual,
                        "reprojection error",
                    );
                });
            if let Some(max) = self.diag.as_ref().map(|d| d.residuals.max.max(1e-6) as f32) {
                ui.separator();
                ui.label("hide above");
                ui.add(
                    egui::Slider::new(&mut self.residual_cutoff, 0.01..=max)
                        .logarithmic(true)
                        .suffix("px"),
                );
                if ui.button("all").clicked() {
                    self.residual_cutoff = max;
                }
            }
            if self.diag.as_ref().is_some_and(|d| d.plane.is_some()) {
                ui.separator();
                ui.checkbox(&mut self.show_plane, "fitted plane");
            }
            if self.scene.is_some() && ui.button("Reset view").clicked() {
                if let Some(s) = &self.scene {
                    self.cam = OrbitCamera::framing(s);
                }
            }
        });
        ui.separator();

        if self.scene.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    self.load_error
                        .clone()
                        .unwrap_or_else(|| "No reconstruction loaded.".into()),
                );
            });
            return;
        }

        let size = egui::vec2(ui.available_width(), height);
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());

        // Orbit / zoom / pan.
        if resp.dragged() {
            let d = resp.drag_delta();
            if resp.dragged_by(egui::PointerButton::Secondary) || ui.input(|i| i.modifiers.shift) {
                // Pan moves the target in the view plane, scaled by
                // distance so it feels the same at any zoom level.
                let r = self.cam.view_rotation();
                let right = r.row(0).transpose();
                let up = r.row(1).transpose();
                let k = self.cam.distance * 0.0015;
                self.cam.target -= right * (d.x as f64 * k);
                self.cam.target += up * (d.y as f64 * k);
            } else {
                self.cam.yaw -= d.x as f64 * 0.008;
                self.cam.pitch = (self.cam.pitch + d.y as f64 * 0.008).clamp(-1.5, 1.5);
            }
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                self.cam.distance = (self.cam.distance * (1.0 - scroll as f64 * 0.0015)).max(1e-4);
            }
        }

        let (w, hpx) = (size.x.max(1.0) as usize, size.y.max(1.0) as usize);
        let dark = ui.visuals().dark_mode;
        let scene = self.scene.as_ref().unwrap();

        // Residual colours and the visibility mask are rebuilt each frame from
        // the cached residuals rather than stored on the app, so the slider and
        // the colour mode stay live without a separate invalidation path. It is
        // one pass over the points - cheaper than the rasterisation that
        // follows it.
        let (mut colors, mut visible) = (None, None);
        if let Some(diag) = &self.diag {
            let max = diag.residuals.max;
            let mut c = Vec::with_capacity(scene.point_ids.len());
            let mut v = Vec::with_capacity(scene.point_ids.len());
            for &id in &scene.point_ids {
                let e = diag.residuals.point_mean.get(&id).copied().unwrap_or(0.0);
                c.push(scene::error_color(e, max));
                v.push(e as f32 <= self.residual_cutoff);
            }
            if self.color_mode == ColorMode::Residual {
                colors = Some(c);
            }
            visible = Some(v);
        }
        let plane_overlay = if self.show_plane {
            self.diag
                .as_ref()
                .and_then(|d| d.plane.as_ref())
                .map(|p| PlaneOverlay {
                    centroid: p.centroid,
                    basis: p.basis,
                    half_extent: scene.extent,
                })
        } else {
            None
        };
        let opts = RenderOptions {
            show_points: self.show_points,
            show_cameras: self.show_cameras,
            point_size: self.point_size,
            bg: if dark { [22, 24, 28] } else { [242, 243, 246] },
            point_colors: colors.as_deref(),
            point_visible: visible.as_deref(),
            plane: plane_overlay.as_ref(),
        };
        let (rgba, projected) = scene::render(scene, &self.cam, w, hpx, &opts);
        let img = egui::ColorImage::from_rgba_unmultiplied([w, hpx], &rgba);
        let tex = self.texture.get_or_insert_with(|| {
            ctx.load_texture("viewport", img.clone(), egui::TextureOptions::LINEAR)
        });
        tex.set(img, egui::TextureOptions::LINEAR);
        painter.image(
            tex.id(),
            resp.rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        let origin = resp.rect.min;
        let to_screen = |p: [f32; 2]| egui::pos2(origin.x + p[0], origin.y + p[1]);

        if !projected.plane_edges.is_empty() {
            let stroke = egui::Stroke::new(
                1.0_f32,
                egui::Color32::from_rgb(130, 130, 150).gamma_multiply(0.5),
            );
            for (a, b) in &projected.plane_edges {
                painter.line_segment([to_screen(*a), to_screen(*b)], stroke);
            }
        }

        if self.show_cameras {
            // Frusta are tinted by their angle to the fitted plane when the
            // scene is planar: the tilt spread decides whether self-calibration
            // was ever going to work, and reading it off the geometry itself
            // beats reading it off a list.
            let tilt_of: BTreeMap<u32, f64> = self
                .diag
                .as_ref()
                .and_then(|d| d.plane.as_ref())
                .filter(|p| p.verdict != CoverageVerdict::NotPlanar)
                .map(|p| p.views.iter().map(|v| (v.image_id, v.tilt_deg)).collect())
                .unwrap_or_default();
            for (i, edges) in &projected.frusta {
                let selected = Some(*i) == self.selected;
                let color = if selected {
                    egui::Color32::from_rgb(255, 170, 40)
                } else if let Some(&tilt) = tilt_of.get(&scene.images[*i].id) {
                    // The ramp runs over 0-60 degrees of tilt away from
                    // fronto-parallel, which is the range Zhang's method cares
                    // about.
                    let c = scene::error_color(tilt, 60.0);
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                } else if dark {
                    egui::Color32::from_rgb(110, 170, 255)
                } else {
                    egui::Color32::from_rgb(30, 90, 200)
                };
                let stroke = egui::Stroke::new(if selected { 2.0_f32 } else { 1.0_f32 }, color);
                for (a, b) in edges {
                    painter.line_segment([to_screen(*a), to_screen(*b)], stroke);
                }
            }
        }

        // Ring the selected point, so one picked from the outlier list is
        // findable in the cloud rather than merely named in the panel.
        if let Some(p) = self.selected_point.and_then(|i| scene.points.get(i)) {
            let eye = self.cam.eye();
            let r = self.cam.view_rotation();
            let v = r * (p.0.cast::<f64>() - eye);
            if v.z > 1e-6 {
                let f = (hpx as f64 * 0.5) / (self.cam.fov_y * 0.5).tan();
                let s = egui::pos2(
                    origin.x + (f * v.x / v.z + w as f64 * 0.5) as f32,
                    origin.y + (f * v.y / v.z + hpx as f64 * 0.5) as f32,
                );
                painter.circle_stroke(
                    s,
                    7.0,
                    egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 170, 40)),
                );
            }
        }

        // A click takes the nearest camera centre within a tolerance, and falls
        // through to picking a 3D point when no camera is close - the frustum
        // is the coarser target, so it wins ties.
        if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let mut best: Option<(f32, usize)> = None;
                for (i, s) in &projected.image_screen {
                    let d = to_screen(*s).distance(pos);
                    if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                        best = Some((d, *i));
                    }
                }
                match best.filter(|(d, _)| *d < 18.0).map(|(_, i)| i) {
                    Some(i) => {
                        self.select_image(i);
                        self.selected_point = None;
                    }
                    None => {
                        let local = pos - origin;
                        self.selected_point =
                            scene::pick_point(&projected, w, hpx, local.x, local.y);
                        if self.selected_point.is_none() {
                            self.selected = None;
                        }
                    }
                }
            }
        }

        if self.color_mode == ColorMode::Residual {
            if let Some(diag) = &self.diag {
                color_legend(&painter, resp.rect, diag.residuals.max);
            }
        }
    }

    fn view_image(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, height: f32) {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_all_keypoints, "all keypoints");
            ui.checkbox(&mut self.show_residual_vectors, "residual vectors");
            ui.label("×");
            ui.add(
                egui::Slider::new(&mut self.residual_exaggeration, 1.0..=200.0).logarithmic(true),
            );
            ui.separator();
            if ui.button("fit").clicked() {
                self.image_zoom = 1.0;
                self.image_pan = egui::Vec2::ZERO;
            }
            if let Some(l) = &self.loaded_image {
                ui.label(
                    egui::RichText::new(l.path.display().to_string())
                        .small()
                        .weak(),
                );
            }
        });
        ui.separator();

        let size = egui::vec2(ui.available_width(), height);
        if self.scene.is_none() || self.recon.is_none() || self.diag.is_none() {
            ui.centered_and_justified(|ui| ui.label("No reconstruction loaded."));
            return;
        }
        let Some(sel) = self.selected else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a camera - in the 3D view, or from the Selection panel.")
            });
            return;
        };
        let (image_id, name) = {
            let scene = self.scene.as_ref().unwrap();
            let v = &scene.images[sel];
            (v.id, v.name.clone())
        };

        if self.loaded_image.as_ref().map(|l| l.image_id) != Some(image_id) {
            match Project::open(&self.project) {
                Ok(p) => match imageview::load(
                    ctx,
                    &self.project,
                    &p.effective_images_dir(),
                    image_id,
                    &name,
                ) {
                    Ok(l) => {
                        self.loaded_image = Some(l);
                        self.image_error = None;
                    }
                    Err(e) => {
                        self.loaded_image = None;
                        self.image_error = Some(e);
                    }
                },
                Err(e) => {
                    self.loaded_image = None;
                    self.image_error = Some(e.to_string());
                }
            }
        }

        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        if self.loaded_image.is_none() {
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                self.image_error.as_deref().unwrap_or("no image"),
                egui::FontId::proportional(13.0),
                ui.visuals().warn_fg_color,
            );
            return;
        }

        // Pan and zoom about the cursor, so inspecting one residual on a 12MP
        // photo does not mean losing the rest of the frame.
        if resp.dragged() {
            self.image_pan += resp.drag_delta();
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let old = self.image_zoom;
                self.image_zoom = (self.image_zoom * (1.0 + scroll * 0.0015)).clamp(0.2, 40.0);
                if let Some(p) = resp.hover_pos() {
                    // Keep whatever is under the cursor fixed while zooming.
                    let c = resp.rect.center() + self.image_pan;
                    self.image_pan += (c - p) * (self.image_zoom / old - 1.0);
                }
            }
        }

        let loaded = self.loaded_image.as_ref().unwrap();
        let src = loaded.source_size;
        let fit = (resp.rect.width() / src[0]).min(resp.rect.height() / src[1]);
        let scale = fit * self.image_zoom;
        let rect = egui::Rect::from_center_size(
            resp.rect.center() + self.image_pan,
            egui::vec2(src[0] * scale, src[1] * scale),
        );
        painter.image(
            loaded.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        // Clip the overlay to the viewport, so a panned-away image does not
        // scribble its keypoints over the rest of the UI.
        let painter = painter.with_clip_rect(resp.rect);
        let to_screen = |x: f64, y: f64| {
            egui::pos2(rect.min.x + x as f32 * scale, rect.min.y + y as f32 * scale)
        };

        let recon = self.recon.as_ref().unwrap();
        let diag = self.diag.as_ref().unwrap();
        let Some(image) = recon.images.get(&image_id) else {
            return;
        };

        let num_triangulated = image.point3d_ids.iter().filter(|p| p.is_some()).count();
        if self.show_all_keypoints {
            // Detected-but-untriangulated keypoints are the whole point of this
            // overlay: a frame densely covered in them says detection worked
            // and matching did not, which no aggregate number distinguishes.
            let untracked = egui::Color32::from_rgb(150, 150, 150).gamma_multiply(0.7);
            for (i, &(x, y)) in image.keypoints.iter().enumerate() {
                if image.point3d_ids.get(i).copied().flatten().is_none() {
                    painter.circle_filled(to_screen(x as f64, y as f64), 1.5, untracked);
                }
            }
        }

        if let Some(idx) = diag.residuals.by_image.get(&image_id) {
            for &i in idx {
                let o = &diag.residuals.all[i];
                let c = scene::error_color(o.residual_px, diag.residuals.max);
                let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
                let m = to_screen(o.measured.0, o.measured.1);
                painter.circle_stroke(m, 2.5, egui::Stroke::new(1.0_f32, color));
                if self.show_residual_vectors {
                    // From the detected keypoint toward where the triangulated
                    // point actually lands, exaggerated: a healthy model's
                    // residuals are a fraction of a pixel and would otherwise
                    // be invisible at any zoom.
                    let k = self.residual_exaggeration as f64;
                    let tip = to_screen(
                        o.measured.0 + (o.projected.0 - o.measured.0) * k,
                        o.measured.1 + (o.projected.1 - o.measured.1) * k,
                    );
                    painter.line_segment([m, tip], egui::Stroke::new(1.0_f32, color));
                }
            }
        }

        let d = diag.images.get(&image_id);
        painter.text(
            resp.rect.left_top() + egui::vec2(6.0, 4.0),
            egui::Align2::LEFT_TOP,
            format!(
                "{name}   {} keypoints, {num_triangulated} triangulated ({:.0}%)   \
                 mean {:.3}px  max {:.3}px",
                image.keypoints.len(),
                if image.keypoints.is_empty() {
                    0.0
                } else {
                    100.0 * num_triangulated as f32 / image.keypoints.len() as f32
                },
                d.map(|d| d.mean_residual_px).unwrap_or(0.0),
                d.map(|d| d.max_residual_px).unwrap_or(0.0),
            ),
            egui::FontId::monospace(11.0),
            ui.visuals().strong_text_color(),
        );
    }

    fn view_graph(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, height: f32) {
        self.ensure_graph(ctx);
        let mut clicked: Option<u32> = None;
        {
            let Ok(state) = self.graph.lock() else {
                ui.label("match graph unavailable");
                return;
            };
            let g = match &*state {
                GraphState::Idle | GraphState::Loading => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("reading verified pairs from the project database...");
                    });
                    return;
                }
                GraphState::Failed(e) => {
                    ui.centered_and_justified(|ui| ui.label(e.as_str()));
                    return;
                }
                GraphState::Ready(g) => g,
            };

            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "{} images  ·  {} verified pairs  ·  {} connected component(s)  ·  {} registered",
                    g.nodes.len(),
                    g.edges.len(),
                    g.components.len(),
                    g.nodes.iter().filter(|n| n.registered).count(),
                ));
            });
            if g.components.len() > 1 {
                let stranded = g
                    .nodes
                    .iter()
                    .filter(|n| n.component > 0 && !n.registered)
                    .count();
                ui.label(
                    egui::RichText::new(format!(
                        "the match graph is split into {} components, and {stranded} image(s) \
                         outside the largest one are unregistered. An incremental reconstruction \
                         grows from one seed and cannot cross a component boundary, so these \
                         cannot register however good their own matches are.",
                        g.components.len()
                    ))
                    .small()
                    .color(warn_color()),
                );
            }
            let isolated = g.isolated();
            if !isolated.is_empty() {
                ui.label(
                    egui::RichText::new(format!(
                        "{} image(s) have no verified pair at all: {}",
                        isolated.len(),
                        isolated
                            .iter()
                            .take(6)
                            .map(|&i| g.nodes[i].name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .small()
                    .color(warn_color()),
                );
            }
            ui.separator();

            let size = egui::vec2(ui.available_width(), (height - 40.0).max(80.0));
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::click());
            let rect = resp.rect;
            let r = rect.width().min(rect.height()) * 0.44;
            let c = rect.center();
            let to_screen = |p: [f32; 2]| egui::pos2(c.x + p[0] * r, c.y + p[1] * r);

            for e in &g.edges {
                let t = e.inliers as f32 / g.max_inliers as f32;
                let col = COMPONENT_COLORS[g.nodes[e.a].component % COMPONENT_COLORS.len()];
                painter.line_segment(
                    [to_screen(g.nodes[e.a].pos), to_screen(g.nodes[e.b].pos)],
                    egui::Stroke::new(0.5 + t * 2.0, col.gamma_multiply(0.15 + t * 0.5)),
                );
            }

            let hover = resp.hover_pos();
            let mut hovered: Option<usize> = None;
            for (i, n) in g.nodes.iter().enumerate() {
                let p = to_screen(n.pos);
                let col = COMPONENT_COLORS[n.component % COMPONENT_COLORS.len()];
                let near = hover.is_some_and(|h| h.distance(p) < 8.0);
                if near {
                    hovered = Some(i);
                }
                // Registered images are filled, unregistered hollow: which
                // images the reconstruction could not reach is the point.
                if n.registered {
                    painter.circle_filled(p, if near { 6.0 } else { 4.0 }, col);
                } else {
                    painter.circle_stroke(
                        p,
                        if near { 6.0 } else { 4.0 },
                        egui::Stroke::new(1.5_f32, col),
                    );
                }
            }
            if let Some(i) = hovered {
                let n = &g.nodes[i];
                painter.text(
                    to_screen(n.pos) + egui::vec2(9.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    format!(
                        "{}  ·  {} pair(s)  ·  component {}{}",
                        n.name,
                        n.degree,
                        n.component,
                        if n.registered {
                            ""
                        } else {
                            "  ·  UNREGISTERED"
                        }
                    ),
                    egui::FontId::monospace(11.0),
                    ui.visuals().strong_text_color(),
                );
                if resp.clicked() {
                    clicked = Some(n.image_id);
                }
            }

            let mut y = rect.top() + 6.0;
            for (i, comp) in g.components.iter().enumerate().take(COMPONENT_COLORS.len()) {
                painter.circle_filled(
                    egui::pos2(rect.left() + 10.0, y + 5.0),
                    4.0,
                    COMPONENT_COLORS[i % COMPONENT_COLORS.len()],
                );
                painter.text(
                    egui::pos2(rect.left() + 20.0, y),
                    egui::Align2::LEFT_TOP,
                    format!("component {i}: {} image(s)", comp.len()),
                    egui::FontId::monospace(11.0),
                    ui.visuals().text_color(),
                );
                y += 14.0;
            }
        }

        if let Some(image_id) = clicked {
            let found = self
                .scene
                .as_ref()
                .and_then(|s| s.images.iter().position(|im| im.id == image_id));
            if let Some(i) = found {
                self.select_image(i);
                self.view = View::Image;
            }
        }
    }

    /// Try the ArUco detector on one real frame and see what it found.
    fn view_aruco(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, height: f32) {
        // Populate the image list from the project the first time in. Uses the
        // effective tree, so a `[layout]` project offers the same images the
        // pipeline would actually read.
        if self.aruco.candidates.is_empty() {
            if let Ok(p) = Project::open(&self.project) {
                if let Ok((imgs, _)) = crate::dataset::discover(&p.effective_images_dir()) {
                    self.aruco.candidates = imgs.into_iter().map(|d| d.path).collect();
                }
            }
        }
        if self.aruco.candidates.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No images found in this project. Open a dataset directory first.")
            });
            return;
        }

        let mut rerun = false;
        ui.horizontal_wrapped(|ui| {
            let n = self.aruco.candidates.len();
            let name = self.aruco.candidates[self.aruco.selected]
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            egui::ComboBox::from_id_salt("arucoimg")
                .selected_text(format!("{name}  ({}/{n})", self.aruco.selected + 1))
                .width(220.0)
                .show_ui(ui, |ui| {
                    // A long list of full paths is unreadable and slow to
                    // scroll; the tail is the part that identifies the frame.
                    for (i, p) in self.aruco.candidates.iter().enumerate().take(500) {
                        let label = p
                            .iter()
                            .rev()
                            .take(3)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<PathBuf>()
                            .display()
                            .to_string();
                        ui.selectable_value(&mut self.aruco.selected, i, label);
                    }
                });
            if ui.button("◀").clicked() && self.aruco.selected > 0 {
                self.aruco.selected -= 1;
                rerun = true;
            }
            if ui.button("▶").clicked() && self.aruco.selected + 1 < n {
                self.aruco.selected += 1;
                rerun = true;
            }
            ui.separator();
            if ui.button("Detect").clicked() {
                rerun = true;
            }
            if ui.button("Reset params").clicked() {
                self.aruco.params = sfm_features::aruco::ArucoParams::default();
                rerun = true;
            }
            ui.checkbox(&mut self.aruco.show_ids, "ids");
            if ui.button("fit").clicked() {
                self.aruco.zoom = 1.0;
                self.aruco.pan = egui::Vec2::ZERO;
            }
        });

        let p = &mut self.aruco.params;
        ui.horizontal_wrapped(|ui| {
            let mut changed = false;
            changed |= ui
                .add(egui::Slider::new(&mut p.adaptive_radius, 1..=41).text("radius"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut p.adaptive_c, 0.0..=60.0).text("threshold C"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.min_component_pixels, 4..=2000)
                        .logarithmic(true)
                        .text("min px"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.min_perimeter_px, 8.0..=400.0)
                        .logarithmic(true)
                        .text("min perimeter"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut p.max_hamming_distance, 0..=6).text("max hamming"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut p.contrast, 0.5..=3.0).text("contrast"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut p.gamma, 0.3..=3.0).text("gamma"))
                .changed();
            ui.separator();
            // Family is detected per frame by default; pinning it is for when
            // a board's markers are too small or blurred to vote reliably.
            let mut pinned = p.data_bits.unwrap_or(0);
            egui::ComboBox::from_id_salt("databits")
                .selected_text(match p.data_bits {
                    None => "family: auto".to_string(),
                    Some(n) => format!("family: {n}x{n}"),
                })
                .width(120.0)
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut pinned, 0, "auto").clicked() {
                        p.data_bits = None;
                        changed = true;
                    }
                    for n in sfm_features::aruco::CANDIDATE_DATA_BITS {
                        if ui
                            .selectable_value(&mut pinned, n, format!("{n}x{n}"))
                            .clicked()
                        {
                            p.data_bits = Some(n);
                            changed = true;
                        }
                    }
                });
            changed |= ui
                .checkbox(&mut p.dictionary_free, "id from pattern")
                .on_hover_text(
                    "off: match this crate's own generated dictionary instead \
                     (only for markers printed from it)",
                )
                .changed();
            // Re-detect on release rather than per-frame: a 12MP detection is
            // ~0.3s, so tracking every intermediate slider value would queue up
            // work the user never asked to see.
            if changed && ui.ctx().input(|i| i.pointer.any_released()) {
                rerun = true;
            }
        });
        ui.separator();

        if rerun {
            self.aruco.spawn(ctx);
        }

        let size = egui::vec2(ui.available_width(), (height - 90.0).max(120.0));
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());

        // Show the source frame under the overlay.
        let path = self.aruco.candidates[self.aruco.selected].clone();
        if self.aruco.texture.as_ref().map(|(p, ..)| p) != Some(&path) {
            if let Ok(img) = image::open(&path) {
                let rgba = img.to_rgba8();
                let src = [rgba.width() as f32, rgba.height() as f32];
                let scale = (2000.0 / rgba.width().max(rgba.height()) as f32).min(1.0);
                let rgba = if scale < 1.0 {
                    image::imageops::resize(
                        &rgba,
                        (rgba.width() as f32 * scale) as u32,
                        (rgba.height() as f32 * scale) as u32,
                        image::imageops::FilterType::Triangle,
                    )
                } else {
                    rgba
                };
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [rgba.width() as usize, rgba.height() as usize],
                    rgba.as_raw(),
                );
                let tex = ctx.load_texture("aruco-src", color, egui::TextureOptions::LINEAR);
                self.aruco.texture = Some((path.clone(), tex, src));
            }
        }

        if resp.dragged() {
            self.aruco.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let old = self.aruco.zoom;
                self.aruco.zoom = (self.aruco.zoom * (1.0 + scroll * 0.0015)).clamp(0.2, 40.0);
                if let Some(pos) = resp.hover_pos() {
                    let c = resp.rect.center() + self.aruco.pan;
                    self.aruco.pan += (c - pos) * (self.aruco.zoom / old - 1.0);
                }
            }
        }

        let Some((_, tex, src)) = &self.aruco.texture else {
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                "could not read that image",
                egui::FontId::proportional(13.0),
                ui.visuals().warn_fg_color,
            );
            return;
        };
        let fit = (resp.rect.width() / src[0]).min(resp.rect.height() / src[1]);
        let scale = fit * self.aruco.zoom;
        let rect = egui::Rect::from_center_size(
            resp.rect.center() + self.aruco.pan,
            egui::vec2(src[0] * scale, src[1] * scale),
        );
        painter.image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let painter = painter.with_clip_rect(resp.rect);
        let to_screen = |x: f32, y: f32| egui::pos2(rect.min.x + x * scale, rect.min.y + y * scale);

        let Ok(state) = self.aruco.state.lock() else {
            return;
        };
        let caption = match &*state {
            aruco::State::Idle => "press Detect".to_string(),
            aruco::State::Running => "detecting...".to_string(),
            aruco::State::Failed(e) => e.clone(),
            aruco::State::Done(d) => {
                for m in &d.markers {
                    let col = aruco::marker_color(m.id);
                    for k in 0..4 {
                        let a = m.corners[k];
                        let b = m.corners[(k + 1) % 4];
                        painter.line_segment(
                            [to_screen(a[0], a[1]), to_screen(b[0], b[1])],
                            egui::Stroke::new(1.5_f32, col),
                        );
                    }
                    // Mark corner 0 so the detected orientation is visible -
                    // a marker found with the wrong rotation still draws a
                    // perfectly good quad.
                    painter.circle_filled(to_screen(m.corners[0][0], m.corners[0][1]), 3.0, col);
                    if self.aruco.show_ids {
                        let cx = m.corners.iter().map(|c| c[0]).sum::<f32>() / 4.0;
                        let cy = m.corners.iter().map(|c| c[1]).sum::<f32>() / 4.0;
                        painter.text(
                            to_screen(cx, cy),
                            egui::Align2::CENTER_CENTER,
                            m.id.to_string(),
                            egui::FontId::monospace(12.0),
                            col,
                        );
                    }
                }
                format!(
                    "{} marker(s), {} corners   ·   family {}   ·   decode {}  detect {}   ·   {:.0}x{:.0}",
                    d.markers.len(),
                    d.num_corners,
                    match d.family {
                        Some(n) => format!("{n}x{n}"),
                        None => "none found".into(),
                    },
                    crate::progress::human_secs(d.decode.as_secs_f64()),
                    crate::progress::human_secs(d.detect.as_secs_f64()),
                    d.source_size[0],
                    d.source_size[1],
                )
            }
        };
        painter.text(
            resp.rect.left_top() + egui::vec2(6.0, 4.0),
            egui::Align2::LEFT_TOP,
            caption,
            egui::FontId::monospace(12.0),
            ui.visuals().strong_text_color(),
        );
    }

    fn view_coverage(&mut self, ui: &mut egui::Ui, height: f32) {
        if self.diag.as_ref().and_then(|d| d.plane.as_ref()).is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded, or too few points to fit a target plane.")
            });
            return;
        }
        ui.label(
            "Where the target plane has been seen from. Each ring is a tilt band away from \
             fronto-parallel; each wedge is the direction of that tilt. Empty wedges are the \
             orientations this capture is still missing.",
        );
        let plane = self.diag.as_ref().unwrap().plane.as_ref().unwrap();
        match &plane.verdict {
            CoverageVerdict::NotPlanar => {
                ui.label(
                    egui::RichText::new(format!(
                        "note: this cloud is not planar (flatness {:.3}), so the plane below is \
                         only a best fit and its coverage is not a calibration criterion",
                        plane.flatness
                    ))
                    .small()
                    .weak(),
                );
            }
            CoverageVerdict::Narrow(msg) => {
                ui.label(egui::RichText::new(msg).small().color(warn_color()));
            }
            CoverageVerdict::Adequate => {
                ui.label(
                    egui::RichText::new("coverage is adequate for single-plane self-calibration")
                        .small()
                        .color(ok_color()),
                );
            }
        }
        ui.separator();
        let size = egui::vec2(ui.available_width(), (height - 60.0).max(120.0));
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
        coverage_plot(&painter, resp.rect, plane, ui.visuals());
    }
}

// ----- small drawing helpers ----------------------------------------------

/// A horizontal bar of the given fraction, used in the observation-count list.
fn bar(ui: &mut egui::Ui, frac: f32, width: f32, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 8.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 1.0, ui.visuals().faint_bg_color);
    let mut filled = rect;
    filled.set_width(rect.width() * frac.clamp(0.0, 1.0));
    ui.painter().rect_filled(filled, 1.0, color);
}

/// Track-length histogram. Its shape is the single most diagnostic picture for
/// self-calibration: a distribution piled entirely at 2 means no point is seen
/// by enough views to constrain a shared focal length, whatever else the
/// numbers say.
fn histogram(ui: &mut egui::Ui, t: &diagnostics::TrackStats, height: f32) {
    if t.histogram.is_empty() {
        return;
    }
    let width = ui.available_width().min(320.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
    let max_count = t.histogram.iter().map(|(_, c)| *c).max().unwrap_or(1) as f32;
    let n = t.histogram.len().max(1);
    let bw = rect.width() / n as f32;
    for (i, (len, count)) in t.histogram.iter().enumerate() {
        let h = (*count as f32 / max_count) * (rect.height() - 14.0);
        let x = rect.left() + i as f32 * bw;
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x + 1.0, rect.bottom() - 14.0 - h),
                egui::pos2(x + bw - 1.0, rect.bottom() - 14.0),
            ),
            1.0,
            // Length-2 tracks carry no multi-view redundancy, so a chart
            // dominated by them is itself the diagnosis.
            if *len <= 2 {
                warn_color()
            } else {
                egui::Color32::from_rgb(90, 140, 220)
            },
        );
        if bw > 14.0 || i == 0 || i + 1 == n {
            painter.text(
                egui::pos2(x + bw * 0.5, rect.bottom() - 12.0),
                egui::Align2::CENTER_TOP,
                format!("{len}"),
                egui::FontId::monospace(9.0),
                ui.visuals().weak_text_color(),
            );
        }
    }
}

/// Every camera's tilt to the fitted plane as a strip of ticks - a compact way
/// to see whether the angles cluster or spread.
fn tilt_strip(ui: &mut egui::Ui, plane: &PlaneDiag) {
    let width = ui.available_width().min(320.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 34.0), egui::Sense::hover());
    let painter = ui.painter();
    let track = egui::Rect::from_min_max(
        egui::pos2(rect.left(), rect.top() + 6.0),
        egui::pos2(rect.right(), rect.top() + 18.0),
    );
    painter.rect_filled(track, 2.0, ui.visuals().faint_bg_color);
    for v in &plane.views {
        let t = (v.tilt_deg / 90.0).clamp(0.0, 1.0) as f32;
        let x = track.left() + t * track.width();
        let c = scene::error_color(v.tilt_deg, 60.0);
        painter.line_segment(
            [egui::pos2(x, track.top()), egui::pos2(x, track.bottom())],
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(c[0], c[1], c[2])),
        );
    }
    for (frac, label) in [(0.0, "0°"), (0.5, "45°"), (1.0, "90°")] {
        painter.text(
            egui::pos2(track.left() + frac * track.width(), track.bottom() + 2.0),
            egui::Align2::CENTER_TOP,
            label,
            egui::FontId::monospace(9.0),
            ui.visuals().weak_text_color(),
        );
    }
}

/// The polar coverage plot: rings are tilt bands, wedges are tilt directions.
fn coverage_plot(
    painter: &egui::Painter,
    rect: egui::Rect,
    plane: &PlaneDiag,
    visuals: &egui::Visuals,
) {
    let grid = diagnostics::coverage_grid(&plane.views);
    let c = rect.center();
    let r_max = rect.width().min(rect.height()) * 0.42;
    let bands = TILT_BANDS.len();
    let max_count = grid
        .iter()
        .flat_map(|b| b.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f32;

    for (b, counts) in grid.iter().enumerate() {
        let r_in = r_max * b as f32 / bands as f32;
        let r_out = r_max * (b + 1) as f32 / bands as f32;
        // The innermost band is fronto-parallel, where azimuth carries no
        // information, so it is one disc rather than eight wedges.
        let sectors = if b == 0 { 1 } else { 8 };
        for s in 0..sectors {
            let count = if b == 0 {
                counts.iter().sum::<usize>()
            } else {
                counts[s]
            };
            let (a0, a1) = if b == 0 {
                (0.0, std::f32::consts::TAU)
            } else {
                (
                    s as f32 / 8.0 * std::f32::consts::TAU,
                    (s + 1) as f32 / 8.0 * std::f32::consts::TAU,
                )
            };
            // egui has no arc primitive, so approximate the wedge with a fan;
            // ten steps is smooth enough at this size.
            const STEPS: usize = 10;
            let mut pts = Vec::with_capacity(STEPS * 2 + 2);
            for k in 0..=STEPS {
                let a = a0 + (a1 - a0) * k as f32 / STEPS as f32;
                pts.push(c + egui::vec2(a.cos(), a.sin()) * r_out);
            }
            for k in (0..=STEPS).rev() {
                let a = a0 + (a1 - a0) * k as f32 / STEPS as f32;
                pts.push(c + egui::vec2(a.cos(), a.sin()) * r_in.max(0.001));
            }
            // Missing orientations are the actionable content, so they are left
            // visibly empty and outlined in the warning colour rather than
            // merely drawn dimmer than the rest.
            let (fill, stroke_color) = if count == 0 {
                (
                    egui::Color32::TRANSPARENT,
                    warn_color().gamma_multiply(0.55),
                )
            } else {
                let t = count as f32 / max_count;
                (
                    egui::Color32::from_rgb(60, 130, 220).gamma_multiply(0.25 + 0.65 * t),
                    visuals.weak_text_color(),
                )
            };
            painter.add(egui::Shape::convex_polygon(
                pts,
                fill,
                egui::Stroke::new(1.0_f32, stroke_color),
            ));
            if count > 0 {
                let am = (a0 + a1) * 0.5;
                painter.text(
                    c + egui::vec2(am.cos(), am.sin()) * (r_in + r_out) * 0.5,
                    egui::Align2::CENTER_CENTER,
                    format!("{count}"),
                    egui::FontId::monospace(11.0),
                    visuals.strong_text_color(),
                );
            }
        }
    }

    for (b, (lo, hi)) in TILT_BANDS.iter().enumerate() {
        painter.text(
            egui::pos2(c.x, c.y - r_max * (b + 1) as f32 / bands as f32 + 7.0),
            egui::Align2::CENTER_CENTER,
            format!("{lo:.0}-{hi:.0}°"),
            egui::FontId::monospace(9.0),
            visuals.weak_text_color(),
        );
    }
    let missing: usize = grid
        .iter()
        .enumerate()
        .map(|(b, counts)| {
            if b == 0 {
                usize::from(counts.iter().sum::<usize>() == 0)
            } else {
                counts.iter().filter(|&&n| n == 0).count()
            }
        })
        .sum();
    let total = 1 + (bands - 1) * 8;
    painter.text(
        rect.left_bottom() + egui::vec2(8.0, -8.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{missing} of {total} orientation cells still empty"),
        egui::FontId::monospace(11.0),
        if missing * 2 > total {
            warn_color()
        } else {
            visuals.text_color()
        },
    );
}

/// Colour key for the residual ramp, drawn in the corner of the 3D view.
fn color_legend(painter: &egui::Painter, rect: egui::Rect, max: f64) {
    const W: f32 = 140.0;
    const STEPS: usize = 60;
    let bar = egui::Rect::from_min_size(
        egui::pos2(rect.right() - W - 12.0, rect.bottom() - 30.0),
        egui::vec2(W, 10.0),
    );
    for i in 0..STEPS {
        let t = i as f64 / (STEPS - 1) as f64;
        let c = scene::error_color(t * max, max);
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(
                    bar.left() + bar.width() * i as f32 / STEPS as f32,
                    bar.top(),
                ),
                egui::vec2(bar.width() / STEPS as f32 + 1.0, bar.height()),
            ),
            0.0,
            egui::Color32::from_rgb(c[0], c[1], c[2]),
        );
    }
    painter.text(
        bar.left_bottom() + egui::vec2(0.0, 1.0),
        egui::Align2::LEFT_TOP,
        "0px",
        egui::FontId::monospace(9.0),
        egui::Color32::GRAY,
    );
    painter.text(
        bar.right_bottom() + egui::vec2(0.0, 1.0),
        egui::Align2::RIGHT_TOP,
        format!("{max:.1}px"),
        egui::FontId::monospace(9.0),
        egui::Color32::GRAY,
    );
}

// ----- project context -----------------------------------------------------

/// The intrinsics as they stood *before* reconstruction.
///
/// `feature`/`init-cam` write them to the project database and `map` never
/// writes them back, so a database row is still the initial guess after a
/// reconstruction - which is what makes an initial-vs-final comparison
/// possible at all. A missing database, missing rows and unreadable rows all
/// come out the same way: no initial guess on record.
fn initial_cameras(project: &Project) -> BTreeMap<u32, sfm_core::CameraModel> {
    let path = project.database_path();
    if !path.exists() {
        return BTreeMap::new();
    }
    crate::db::Database::open(&path)
        .and_then(|db| db.list_cameras())
        .map(|cams| cams.into_iter().map(|c| (c.camera_id, c.model)).collect())
        .unwrap_or_default()
}

/// Camera ids the config declared `refine = false`, resolved the way `cmd_map`
/// resolves them: by matching each declared glob against the names of the
/// images that ended up on each camera.
fn pinned_cameras(project: &Project, recon: &Reconstruction) -> BTreeSet<u32> {
    let mut pinned = BTreeSet::new();
    for cfg in &project.config.cameras {
        if cfg.refine != Some(false) {
            continue;
        }
        for im in recon.images.values() {
            if crate::project::glob_match(&cfg.images, &im.name) {
                pinned.insert(im.camera_id);
            }
        }
    }
    pinned
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // A finished stage may have produced a new model.
        if self.run.take_finished() {
            self.reload();
        }
        if self.run.is_running() {
            // Keep the log and spinner live while a stage runs.
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        self.top_bar(ctx);
        self.left_panel(ctx);
        self.right_panel(ctx);
        self.central(ctx);
    }
}
