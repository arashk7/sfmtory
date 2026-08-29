//! `sfmtory gui` - a viewer and front-end over the same pipeline the CLI runs.
//!
//! The CLI remains the primary interface: every button here shells out to the
//! same subcommand you would type, against the same project directory, so
//! nothing is reachable through the GUI that is not reachable without it.

mod pipeline;
mod scene;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use eframe::egui;

use crate::project::Project;
use pipeline::RunState;
use scene::{OrbitCamera, RenderOptions, Scene};

pub fn launch(project_dir: PathBuf) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("sfmtory"),
        ..Default::default()
    };
    eframe::run_native(
        "sfmtory",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(project_dir)))),
    )
    .map_err(|e| anyhow::anyhow!("could not start the viewer: {e}"))
}

struct App {
    project: PathBuf,
    scene: Option<Scene>,
    load_error: Option<String>,
    cam: OrbitCamera,
    selected: Option<usize>,
    show_points: bool,
    show_cameras: bool,
    point_size: i32,
    texture: Option<egui::TextureHandle>,
    run: RunState,
    expanded: BTreeSet<PathBuf>,
    show_log: bool,
    /// Stage options mirrored from the CLI's own defaults.
    detector: usize,
    pairing: usize,
    pipeline_kind: usize,
    export_format: usize,
}

const DETECTORS: [&str; 4] = ["sift", "aruco", "orb", "disk"];
const PAIRINGS: [&str; 3] = ["exhaustive", "sequential", "vocab-tree"];
const PIPELINES: [&str; 2] = ["incremental", "global"];
const EXPORTS: [&str; 2] = ["colmap-text", "nerf-transforms"];

impl App {
    fn new(project: PathBuf) -> Self {
        let mut app = App {
            project,
            scene: None,
            load_error: None,
            cam: OrbitCamera {
                target: nalgebra::Vector3::zeros(),
                distance: 5.0,
                yaw: 0.6,
                pitch: 0.35,
                fov_y: 50f64.to_radians(),
            },
            selected: None,
            show_points: true,
            show_cameras: true,
            point_size: 2,
            texture: None,
            run: RunState::default(),
            expanded: BTreeSet::new(),
            show_log: true,
            detector: 0,
            pairing: 0,
            pipeline_kind: 0,
            export_format: 0,
        };
        app.reload();
        app
    }

    /// Loads whatever `sfmtory map` last wrote, if anything.
    fn reload(&mut self) {
        self.load_error = None;
        let dir = match Project::open(&self.project) {
            Ok(p) => p.sparse_dir(),
            Err(e) => {
                self.load_error = Some(format!("{e}"));
                return;
            }
        };
        if !dir.join("cameras.txt").exists() {
            self.scene = None;
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
                self.scene = Some(s);
            }
            Err(e) => {
                self.scene = None;
                self.load_error = Some(format!("failed to read {}: {e}", dir.display()));
            }
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("stages").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                let busy = self.run.is_running();
                ui.strong("Pipeline:");
                ui.add_enabled_ui(!busy, |ui| {
                    egui::ComboBox::from_id_salt("detector")
                        .selected_text(DETECTORS[self.detector])
                        .width(70.0)
                        .show_ui(ui, |ui| {
                            for (i, d) in DETECTORS.iter().enumerate() {
                                ui.selectable_value(&mut self.detector, i, *d);
                            }
                        });
                    if ui.button("① Feature").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "feature".into(),
                                "--detector".into(),
                                DETECTORS[self.detector].into(),
                            ],
                        );
                    }
                    egui::ComboBox::from_id_salt("pairing")
                        .selected_text(PAIRINGS[self.pairing])
                        .width(95.0)
                        .show_ui(ui, |ui| {
                            for (i, p) in PAIRINGS.iter().enumerate() {
                                ui.selectable_value(&mut self.pairing, i, *p);
                            }
                        });
                    if ui.button("② Match").clicked() {
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
                    if ui.button("③ Map").clicked() {
                        self.run.spawn(
                            &self.project,
                            vec![
                                "map".into(),
                                "--pipeline".into(),
                                PIPELINES[self.pipeline_kind].into(),
                            ],
                        );
                    }
                    ui.separator();
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
                    if ui.button("Eval").clicked() {
                        self.run.spawn(&self.project, vec!["eval".into()]);
                    }
                    if ui.button("Init-cam").clicked() {
                        self.run.spawn(&self.project, vec!["init-cam".into()]);
                    }
                });
                ui.separator();
                if ui.button("⟳ Reload model").clicked() {
                    self.reload();
                }
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
                egui::ScrollArea::vertical()
                    .id_salt("tree")
                    .show(ui, |ui| {
                        let root = self.project.clone();
                        self.dir_tree(ui, &root, 0);
                    });
            });
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
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading("Properties");
                ui.separator();
                let Some(scene) = &self.scene else {
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

                let Some(sel) = self.selected else {
                    ui.label("Select a camera in the 3D view, or from the list below.");
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().id_salt("imglist").show(ui, |ui| {
                        for (i, im) in scene.images.iter().enumerate() {
                            if ui.selectable_label(false, &im.name).clicked() {
                                self.selected = Some(i);
                            }
                        }
                    });
                    return;
                };
                let im = &scene.images[sel];
                ui.strong(&im.name);
                ui.label(format!("image id {}  ·  camera id {}", im.id, im.camera_id));
                ui.label(format!("{} observations", im.num_observations));
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
                egui::ScrollArea::vertical()
                    .id_salt("siblings")
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for (i, other) in scene.images.iter().enumerate() {
                            if other.camera_id != cam_id {
                                continue;
                            }
                            if ui
                                .selectable_label(i == sel, &other.name)
                                .clicked()
                            {
                                self.selected = Some(i);
                            }
                        }
                    });
            });
    }

    fn viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_points, "3D points");
                ui.checkbox(&mut self.show_cameras, "Cameras");
                ui.separator();
                ui.label("point size");
                ui.add(egui::Slider::new(&mut self.point_size, 1..=5).show_value(false));
                ui.separator();
                ui.checkbox(&mut self.show_log, "Log");
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

            let avail = ui.available_size();
            let h = if self.show_log {
                (avail.y - 160.0).max(120.0)
            } else {
                avail.y
            };
            let size = egui::vec2(avail.x, h);
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());

            // Orbit / zoom / pan.
            if resp.dragged() {
                let d = resp.drag_delta();
                if resp.dragged_by(egui::PointerButton::Secondary)
                    || ui.input(|i| i.modifiers.shift)
                {
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
                    self.cam.pitch = (self.cam.pitch + d.y as f64 * 0.008)
                        .clamp(-1.5, 1.5);
                }
            }
            if resp.hovered() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll != 0.0 {
                    self.cam.distance =
                        (self.cam.distance * (1.0 - scroll as f64 * 0.0015)).max(1e-4);
                }
            }

            let (w, hpx) = (size.x.max(1.0) as usize, size.y.max(1.0) as usize);
            let dark = ui.visuals().dark_mode;
            let opts = RenderOptions {
                show_points: self.show_points,
                show_cameras: self.show_cameras,
                point_size: self.point_size,
                bg: if dark { [22, 24, 28] } else { [242, 243, 246] },
            };
            let scene = self.scene.as_ref().unwrap();
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

            if self.show_cameras {
                for (i, edges) in &projected.frusta {
                    let selected = Some(*i) == self.selected;
                    let color = if selected {
                        egui::Color32::from_rgb(255, 170, 40)
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

            // Click selects the nearest camera centre, within a tolerance so a
            // click on empty space clears the selection instead of snapping to
            // something far away.
            if resp.clicked() {
                if let Some(pos) = resp.interact_pointer_pos() {
                    let mut best: Option<(f32, usize)> = None;
                    for (i, s) in &projected.image_screen {
                        let d = to_screen(*s).distance(pos);
                        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                            best = Some((d, *i));
                        }
                    }
                    self.selected = best.filter(|(d, _)| *d < 18.0).map(|(_, i)| i);
                }
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
        self.viewport(ctx);
    }
}
