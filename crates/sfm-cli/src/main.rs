mod aruco_tuning;
mod dataset;
mod db;
// Kept out of `gui` and compiled unconditionally: it is plain geometry over a
// reconstruction, so it stays testable (and reusable by the CLI) in a headless
// `--no-default-features` build that has no windowing stack at all. Its only
// caller today is the viewer, hence the allow - the unit tests still run and
// still guard the maths in both builds.
#[cfg_attr(not(feature = "gui"), allow(dead_code))]
mod diagnostics;
#[cfg(feature = "gui")]
mod gui;
mod initcam;
mod layout;
mod project;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use nalgebra::Vector3;
use rayon::prelude::*;
use sfm_core::{Camera, CameraModel};

use db::Database;
use project::Project;

#[derive(Parser)]
#[command(
    name = "sfmtory",
    version,
    about = "Advanced sparse structure-from-motion / camera calibration."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scaffold a new project directory (sfm.toml, database, sparse/, export/, logs/).
    Project {
        #[command(subcommand)]
        action: ProjectAction,
    },
    /// Inspect and normalise the input image tree declared by `[layout]`.
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },
    /// Open the viewer: 3D scene, camera inspection, and the pipeline stages.
    #[cfg(feature = "gui")]
    Gui(GuiArgs),
    /// Stage 0: estimate camera intrinsics before reconstruction.
    #[command(name = "init-cam")]
    InitCam(InitCamArgs),
    /// Stage 1: detect keypoints/descriptors for every image into the project cache.
    #[command(alias = "extract")]
    Feature(FeatureArgs),
    /// Stage 2: pair images and match+geometrically-verify their features.
    Match(MatchArgs),
    /// Stage 3: run sparse reconstruction (poses + 3D points) from verified matches.
    Map(MapArgs),
    /// Stage 3b: standalone bundle-adjustment / re-triangulation refinement pass.
    Refine(RefineArgs),
    /// Stage 4: export the sparse model to COLMAP text or NeRF transforms.json.
    Export(ExportArgs),
    /// Compare a reconstruction against a baseline (e.g. COLMAP) or ground truth.
    Eval(EvalArgs),
    /// Convenience wrapper: extract -> match -> map -> export in one call.
    Run(RunArgs),
    /// Hidden diagnostic: show raw putative + RANSAC stats for one image pair
    /// regardless of the accept/reject threshold.
    #[command(hide = true)]
    DebugPair(DebugPairArgs),
}

#[derive(clap::Args)]
struct DebugPairArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long)]
    image1: u32,
    #[arg(long)]
    image2: u32,
}

#[derive(Subcommand)]
enum ProjectAction {
    New {
        /// Project directory to create.
        dir: PathBuf,
        /// Directory containing the input images.
        #[arg(long)]
        images: PathBuf,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// Build the normalised `capture_<n>/cam<n>/` symlink tree that the
    /// pipeline reads, from the raw tree and the `[layout]` in sfm.toml.
    Link(DatasetLinkArgs),
}

#[derive(clap::Args)]
struct DatasetLinkArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Report the mapping without creating anything.
    #[arg(long)]
    dry_run: bool,
    /// Replace an existing link tree.
    #[arg(long)]
    force: bool,
}

#[cfg(feature = "gui")]
#[derive(clap::Args)]
struct GuiArgs {
    /// Project directory to open. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Which view to open on. Handy for going straight to the diagnostic that
    /// prompted opening the viewer at all - typically the match graph, after a
    /// run left images unregistered.
    #[arg(long, value_enum, default_value_t = GuiViewArg::Scene)]
    view: GuiViewArg,
}

#[cfg(feature = "gui")]
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum GuiViewArg {
    /// The 3D point cloud and camera frusta.
    Scene,
    /// The selected image with its features and residuals drawn on it.
    Image,
    /// Images as nodes, verified pairs as edges, components coloured.
    Graph,
    /// Board-orientation coverage for a planar capture.
    Coverage,
}

#[derive(clap::Args)]
struct InitCamArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// How many images to sample. Intrinsics are a property of the camera, not
    /// of any one frame, so a spread across the dataset is as informative as
    /// all of it and far cheaper.
    #[arg(long, default_value_t = 12)]
    samples: u32,
    /// Also write the chosen camera into the project's `sfm.toml`, so later
    /// stages pick it up with no further action.
    #[arg(long)]
    apply: bool,
}

#[derive(clap::Args)]
struct FeatureArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = DetectorArg::Sift)]
    detector: DetectorArg,
    #[arg(long)]
    aruco_dict: Option<String>,
    #[arg(long)]
    max_features: Option<u32>,
    /// Search ArUco detector and image-preprocessing parameters for ones that
    /// work best on this dataset, then save them to the project so later runs
    /// reuse them automatically.
    #[arg(long)]
    find_params: bool,
    /// Merge each camera's features across captures into a single feature set
    /// per (camera, image slot). For a rig of fixed cameras photographing a
    /// target that moves between captures, this is what turns N captures into
    /// N times the observations of one unmoved viewpoint.
    #[arg(long)]
    merge_multicaps: bool,
    #[arg(long)]
    gpu: bool,
}

// clap needs a concrete default-capable wrapper since `Detector` has no Default.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum DetectorArg {
    Sift,
    Akaze,
    Orb,
    Superpoint,
    Disk,
    Aruco,
}

#[derive(clap::Args)]
struct MatchArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = PairingArg::Sequential)]
    pairing: PairingArg,
    #[arg(long, value_enum, default_value_t = MatcherArg::MnnRatio)]
    matcher: MatcherArg,
    #[arg(long, default_value_t = 10)]
    window: u32,
    #[arg(long)]
    gpu: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum PairingArg {
    Exhaustive,
    Sequential,
    Spatial,
    VocabTree,
    Aruco,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum MatcherArg {
    MnnRatio,
    Lightglue,
}

#[derive(clap::Args)]
struct MapArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = PipelineArg::Global)]
    pipeline: PipelineArg,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum PipelineArg {
    Global,
    Incremental,
}

#[derive(clap::Args)]
struct RefineArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = RobustLossArg::Huber)]
    robust_loss: RobustLossArg,
    #[arg(long)]
    refine_intrinsics: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum RobustLossArg {
    Huber,
    Cauchy,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Output format. Defaults to COLMAP text.
    #[arg(long, value_enum, default_value_t = ExportFormatArg::ColmapText)]
    format: ExportFormatArg,
    /// Destination. Defaults to the project's `export/` directory.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ExportFormatArg {
    ColmapText,
    NerfTransforms,
}

#[derive(clap::Args)]
struct EvalArgs {
    /// COLMAP-format model directory to evaluate (`cameras.txt`,
    /// `images.txt`, `points3D.txt`). Defaults to this project's map output.
    #[arg(long)]
    ours: Option<PathBuf>,
    /// Project directory, used to locate `--ours` when it isn't given.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// A second COLMAP model to compare against (e.g. a real COLMAP run).
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Ground-truth focal length: either a number in pixels, or a path to a
    /// file whose first token is the focal length (a 3x3 K matrix works).
    #[arg(long)]
    gt_focal: Option<String>,
    /// Ground-truth model directory, whose camera focal lengths are averaged
    /// to form the reference instead of `--gt-focal`.
    #[arg(long)]
    gt: Option<PathBuf>,
}

#[derive(clap::Args)]
struct RunArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = DetectorArg::Sift)]
    detector: DetectorArg,
    #[arg(long, value_enum, default_value_t = PairingArg::Sequential)]
    pairing: PairingArg,
    #[arg(long, value_enum, default_value_t = MatcherArg::MnnRatio)]
    matcher: MatcherArg,
    #[arg(long, value_enum, default_value_t = PipelineArg::Global)]
    pipeline: PipelineArg,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Project { action } => match action {
            ProjectAction::New { dir, images } => cmd_project_new(&dir, &images),
        },
        Commands::Dataset { action } => match action {
            DatasetAction::Link(args) => cmd_dataset_link(args),
        },
        #[cfg(feature = "gui")]
        Commands::Gui(args) => gui::launch(args.project, args.view.into()),
        Commands::InitCam(args) => cmd_init_cam(args),
        Commands::Feature(args) => cmd_feature(args),
        Commands::Match(args) => cmd_match(args),
        Commands::Map(args) => cmd_map(args),
        Commands::Refine(args) => cmd_refine(args),
        Commands::Export(args) => cmd_export(args),
        Commands::Eval(args) => cmd_eval(args),
        Commands::Run(args) => cmd_run(args),
        Commands::DebugPair(args) => cmd_debug_pair(args),
    }
}

fn cmd_project_new(dir: &PathBuf, images: &PathBuf) -> Result<()> {
    if !images.is_dir() {
        bail!("images directory does not exist: {}", images.display());
    }
    let project = Project::create(dir, images)?;
    println!(
        "Created project at {} (images: {})",
        project.root.display(),
        project.config.images_dir.display()
    );
    println!("Next: sfmtory feature --project {}", project.root.display());
    Ok(())
}

/// Starting intrinsics for a declared camera with no `params` given: the same
/// wide-FOV focal guess the resolution-grouped path uses, expressed in
/// whichever model was requested. Distortion terms start at zero.
fn default_camera_model(name: &str, w: u32, h: u32) -> Option<CameraModel> {
    let f = w.max(h) as f64 * 1.2;
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let params: Vec<f64> = match name {
        "SIMPLE_PINHOLE" => vec![f, cx, cy],
        "PINHOLE" => vec![f, f, cx, cy],
        "SIMPLE_RADIAL" => vec![f, cx, cy, 0.0],
        "RADIAL" => vec![f, cx, cy, 0.0, 0.0],
        "OPENCV" => vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        "OPENCV_FISHEYE" => vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        _ => return None,
    };
    CameraModel::from_name_and_params(name, &params)
}

fn cmd_dataset_link(args: DatasetLinkArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let Some(cfg) = project.config.layout.clone() else {
        bail!(
            "{} declares no [layout], so there is nothing to normalise.\n\
             Add one when the dataset's directory shape cannot be inferred - for a rig \
             dumping one directory per capture and one file per camera:\n\n\
             [layout]\n\
             capture = \"dir\"\n\
             camera = \"stem\"\n",
            Project::config_path(&project.root).display()
        );
    };
    let source = cfg.source_dir(&project.root, &project.config.images_dir);
    let plan = layout::plan(&source, &cfg)?;

    println!("Source {}", source.display());
    println!(
        "  {} captures x {} cameras -> {} images",
        plan.captures.len(),
        plan.cameras.len(),
        plan.placed.len()
    );
    let full = plan.captures.len() * plan.cameras.len();
    if plan.placed.len() != full {
        println!(
            "  {} of {} (capture, camera) slots are filled",
            plan.placed.len(),
            full
        );
    }
    // A camera missing from some captures is worth naming: it is exactly the
    // multi-capture redundancy the layout exists to provide, quietly reduced.
    if !plan.gaps.is_empty() {
        println!(
            "  {} camera(s) are absent from at least one capture:",
            plan.gaps.len()
        );
        for (cam, missing) in plan.gaps.iter().take(10) {
            println!("    camera {cam} missing from capture(s) {missing:?}");
        }
        if plan.gaps.len() > 10 {
            println!("    ... and {} more", plan.gaps.len() - 10);
        }
    }
    for p in plan.placed.iter().take(3) {
        println!("  e.g. {} -> {}", p.source.display(), p.link.display());
    }

    if args.dry_run {
        println!("\nDry run: nothing written. Re-run without --dry-run to build the tree.");
        return Ok(());
    }

    let target = project.linked_images_dir();
    let n = layout::apply(&plan, &target, args.force)?;
    println!("\nLinked {n} image(s) into {}", target.display());
    println!("Stages will now read that tree; re-run this after the raw dataset changes.");

    let payload = serde_json::json!({
        "stage": "dataset-link",
        "status": "ok",
        "source": source.display().to_string(),
        "target": target.display().to_string(),
        "num_captures": plan.captures.len(),
        "num_cameras": plan.cameras.len(),
        "num_images": plan.placed.len(),
        "num_cameras_with_gaps": plan.gaps.len(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    project.record_log("dataset-link", &payload)?;
    Ok(())
}

fn cmd_init_cam(args: InitCamArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let stage_dir = project.prepare_stage("init-cam")?;

    let (discovered, _layout) = dataset::discover(&project.require_images_dir()?)?;
    let stride = (discovered.len() / (args.samples as usize).max(1)).max(1);
    let picked: Vec<&dataset::DiscoveredImage> = discovered
        .iter()
        .step_by(stride)
        .take(args.samples as usize)
        .collect();
    println!(
        "Estimating intrinsics from {} of {} images in {}",
        picked.len(),
        discovered.len(),
        project.effective_images_dir().display()
    );

    let mut samples = Vec::new();
    for d in &picked {
        let bytes = std::fs::read(&d.path)
            .with_context(|| format!("reading {}", d.path.display()))?;
        let img = image::open(&d.path)
            .with_context(|| format!("decoding {}", d.path.display()))?;
        let (w, h) = (img.width(), img.height());
        samples.push((bytes, img.to_luma8(), w, h));
    }

    // Reuse fiducial detections if `sfmtory feature --detector aruco` has
    // already run - that is where the strongest estimator's input comes from.
    let db_path = project.database_path();
    let mut owned: Vec<(sfm_core::FeatureSet, u32, u32)> = Vec::new();
    if db_path.exists() {
        if let Ok(db) = Database::open(&db_path) {
            if let Ok(images) = db.list_images() {
                for (id, _cam, _name, w, h) in images {
                    if let Ok(fs) = db.load_features(id) {
                        if matches!(fs.descriptors, sfm_core::Descriptors::MarkerCorner { .. }) {
                            owned.push((fs, w, h));
                        }
                    }
                }
            }
        }
    }
    let features: Vec<(&sfm_core::FeatureSet, u32, u32)> =
        owned.iter().map(|(f, w, h)| (f, *w, *h)).collect();
    if !features.is_empty() {
        println!("Using fiducial detections from {} image(s)", features.len());
    }

    let result = initcam::run_cascade(&samples, &features)?;

    println!("\nEstimates (best first):");
    for e in &result.estimates {
        let mark = if e.method == result.method { "->" } else { "  " };
        println!(
            "  {mark} {:<17} f = {:>9.2} px  [{:?}]\n       {}",
            e.method, e.focal_px, e.confidence, e.detail
        );
    }
    if let Some(a) = &result.agreement {
        println!("\n  corroboration: {a}");
    }
    println!(
        "\nSelected: f = {:.2} px via `{}` (confidence {:?})",
        result.focal_px, result.method, result.confidence
    );
    if result.confidence <= initcam::Confidence::Low {
        println!(
            "  NOTE: this is a weak estimate. Supply known intrinsics if you have them - see\n\
             \x20       `sfmtory init-cam --help` and the Camera setup section of the README."
        );
    }

    initcam::write_result(&stage_dir, &result)?;
    println!("\nWrote {} and cameras.toml", stage_dir.join("intrinsics.json").display());

    if args.apply {
        let cfg_path = Project::config_path(&project.root);
        let existing = std::fs::read_to_string(&cfg_path).unwrap_or_default();
        if existing.contains("[[cameras]]") {
            bail!(
                "{} already declares [[cameras]]; not overwriting. Merge \n{}\nby hand instead.",
                cfg_path.display(),
                stage_dir.join("cameras.toml").display()
            );
        }
        let images_line = if existing.trim().is_empty() {
            format!("images_dir = {:?}\n\n", project.config.images_dir)
        } else {
            String::new()
        };
        std::fs::write(
            &cfg_path,
            format!("{existing}{images_line}\n{}", result.as_camera_toml()),
        )?;
        println!("Applied to {}", cfg_path.display());
    } else {
        println!("Re-run with --apply to write this camera into sfm.toml.");
    }

    let payload = serde_json::json!({
        "stage": "init-cam",
        "status": "ok",
        "focal_px": result.focal_px,
        "method": result.method,
        "confidence": format!("{:?}", result.confidence),
        "num_sampled": picked.len(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    project.record_log("init-cam", &payload)?;
    Ok(())
}

fn cmd_feature(args: FeatureArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let stage_dir = project.prepare_stage("feature")?;
    std::fs::create_dir_all(project.cache_dir())?;

    let detector_kind = match args.detector {
        DetectorArg::Sift => sfm_features::DetectorKind::Sift,
        DetectorArg::Orb => sfm_features::DetectorKind::Orb,
        DetectorArg::Aruco => sfm_features::DetectorKind::Aruco,
        DetectorArg::Disk => sfm_features::DetectorKind::Disk,
        other => bail!(
            "detector {other:?} is not implemented yet (see PLAN.md); available now: sift, orb, aruco, disk"
        ),
    };
    if args.gpu && args.detector != DetectorArg::Disk {
        eprintln!("note: --gpu has no effect for {:?} (classical CPU detector); GPU is used for the learned `disk` detector", args.detector);
    }

    let (discovered, layout) = dataset::discover(&project.require_images_dir()?)?;
    let num_captures = discovered
        .iter()
        .map(|d| d.capture_id)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let num_phys_cameras = discovered
        .iter()
        .map(|d| d.camera_id)
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!(
        "Found {} images ({:?} layout: {num_captures} capture(s), {num_phys_cameras} camera(s)) in {}",
        discovered.len(),
        layout,
        project.effective_images_dir().display()
    );

    let mut config = match args.max_features {
        Some(n) => {
            sfm_features::DetectorConfig::new(detector_kind).with_max_features(Some(n as usize))
        }
        None => sfm_features::DetectorConfig::new(detector_kind),
    };
    config.disk.use_gpu = args.gpu;

    // ArUco parameter handling: search on request, otherwise reuse whatever a
    // previous search saved, otherwise defaults.
    if detector_kind == sfm_features::DetectorKind::Aruco {
        if args.find_params {
            let best = aruco_tuning::find_params(&discovered)?;
            aruco_tuning::save(&project.aruco_params_path(), &best)?;
            println!(
                "Saved tuned ArUco parameters to {}",
                project.aruco_params_path().display()
            );
            config.aruco = best;
        } else if let Some(saved) = aruco_tuning::load(&project.aruco_params_path())? {
            println!(
                "Using tuned ArUco parameters from {}",
                project.aruco_params_path().display()
            );
            config.aruco = saved;
        }
    } else if args.find_params {
        bail!("--find-params applies to `--detector aruco` only");
    }

    // Decoding + detection is embarrassingly parallel and touches no shared
    // state; only the sqlite writes afterward need to be sequential. Bounded
    // pool width for the same memory reason documented on the detectors.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads().min(4))
        .build()
        .context("building feature-extraction thread pool")?;
    let detected: Vec<Result<(u32, u32, sfm_core::FeatureSet)>> = pool.install(|| {
        discovered
            .par_iter()
            .map(|d| {
                let (w, h) = image::image_dimensions(&d.path)
                    .with_context(|| format!("reading dimensions of {}", d.path.display()))?;
                let mut features = sfm_features::detect_file(&d.path, &config)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .with_context(|| format!("detecting features in {}", d.path.display()))?;
                // Stamp the capture so a fiducial that moved between captures
                // cannot match itself across them.
                sfm_features::aruco::stamp_capture_id(&mut features, d.capture_id);
                Ok((w, h, features))
            })
            .collect()
    });

    let mut records: Vec<(dataset::DiscoveredImage, u32, u32, sfm_core::FeatureSet)> = Vec::new();
    for (d, r) in discovered.into_iter().zip(detected) {
        let (w, h, f) = r?;
        records.push((d, w, h, f));
    }

    if args.merge_multicaps {
        records = merge_across_captures(records, num_captures)?;
    }

    let db = Database::open(&project.database_path())?;
    let mut camera_by_dims: HashMap<(u32, u32), u32> = HashMap::new();
    let mut next_camera_id = db.max_camera_id()? + 1;
    let mut next_image_id = db.max_image_id()? + 1;
    let declared = &project.config.cameras;
    let mut declared_ids: HashMap<usize, u32> = HashMap::new();
    let mut phys_camera_ids: HashMap<u32, u32> = HashMap::new();
    let mut per_image = Vec::new();
    let mut total_features = 0usize;
    let mut corner_rows: Vec<String> = Vec::new();

    for (d, w, h, features) in &records {
        let (w, h) = (*w, *h);
        // Camera assignment, most explicit source first: declared `[[cameras]]`
        // globs, then the physical camera the layout identified, then image
        // resolution. The middle case is what keeps several same-resolution
        // cameras in a rig from collapsing into one intrinsics block.
        let camera_id = if !declared.is_empty() {
            let which = declared
                .iter()
                .position(|c| project::glob_match(&c.images, &d.name))
                .with_context(|| {
                    format!(
                        "image {} matches none of the {} declared [[cameras]] globs in sfm.toml \
                         (add a catch-all `images = \"*\"` entry if that is intended)",
                        d.name,
                        declared.len()
                    )
                })?;
            if let Some(&id) = declared_ids.get(&which) {
                id
            } else {
                let cfg = &declared[which];
                let id = next_camera_id;
                next_camera_id += 1;
                let model_name = cfg.model.as_deref().unwrap_or("SIMPLE_RADIAL");
                let model = match &cfg.params {
                    Some(params) => CameraModel::from_name_and_params(model_name, params)
                        .with_context(|| {
                            format!(
                                "camera \"{}\": model {model_name} does not accept {} parameters",
                                cfg.name,
                                params.len()
                            )
                        })?,
                    None => default_camera_model(model_name, w, h).with_context(|| {
                        format!("camera \"{}\": unknown model {model_name}", cfg.name)
                    })?,
                };
                db.upsert_camera(&Camera { camera_id: id, model, width: w, height: h })?;
                eprintln!(
                    "camera {id} \"{}\" ({model_name}{}){}",
                    cfg.name,
                    if cfg.params.is_some() { ", known intrinsics" } else { "" },
                    if cfg.refine == Some(false) { ", held fixed" } else { "" }
                );
                declared_ids.insert(which, id);
                id
            }
        } else if num_phys_cameras > 1 {
            match phys_camera_ids.get(&d.camera_id) {
                Some(&id) => id,
                None => {
                    let id = next_camera_id;
                    next_camera_id += 1;
                    db.upsert_camera(&Camera {
                        camera_id: id,
                        model: default_camera_model("SIMPLE_RADIAL", w, h).unwrap(),
                        width: w,
                        height: h,
                    })?;
                    phys_camera_ids.insert(d.camera_id, id);
                    id
                }
            }
        } else if let Some(id) = camera_by_dims.get(&(w, h)) {
            *id
        } else if let Some(id) = db.find_camera_id_by_dims(w, h)? {
            camera_by_dims.insert((w, h), id);
            id
        } else {
            let id = next_camera_id;
            next_camera_id += 1;
            db.upsert_camera(&Camera {
                camera_id: id,
                model: default_camera_model("SIMPLE_RADIAL", w, h).unwrap(),
                width: w,
                height: h,
            })?;
            camera_by_dims.insert((w, h), id);
            id
        };

        let image_id = match db.find_image_id_by_name(&d.name)? {
            Some(id) => id,
            None => {
                let id = next_image_id;
                next_image_id += 1;
                id
            }
        };
        // A merged row spans captures, so it has no single capture of origin.
        let capture_col = if args.merge_multicaps && num_captures > 1 {
            -1
        } else {
            d.capture_id as i64
        };
        db.upsert_image(
            image_id,
            camera_id,
            &d.name,
            w,
            h,
            capture_col,
            d.camera_id,
            d.image_index,
        )?;
        db.store_features(
            image_id,
            &format!("{:?}", args.detector).to_lowercase(),
            features,
        )?;

        // Every fiducial corner gets the identity the spec calls for:
        // capture_camera_image_aruco_corner.
        for i in 0..features.len() {
            if let Some((capture, marker, corner)) = features.descriptors.marker_corner(i) {
                let kp = features.keypoints[i];
                corner_rows.push(format!(
                    "{}_{}_{}_{}_{},{},{},{:.3},{:.3}",
                    capture, d.camera_id, d.image_index, marker, corner,
                    image_id, d.name, kp.x, kp.y
                ));
            }
        }

        total_features += features.len();
        per_image.push(serde_json::json!({
            "image_id": image_id,
            "name": d.name,
            "capture_id": capture_col,
            "camera_id": d.camera_id,
            "image_index": d.image_index,
            "num_features": features.len(),
        }));
    }

    if !corner_rows.is_empty() {
        let path = stage_dir.join("corners.csv");
        let mut out =
            String::from("feature_id,image_id,image_name,x,y\n");
        out.push_str(&corner_rows.join("\n"));
        out.push('\n');
        std::fs::write(&path, out)
            .with_context(|| format!("writing {}", path.display()))?;
        println!("Wrote {} fiducial corner ids to {}", corner_rows.len(), path.display());
    }

    let payload = serde_json::json!({
        "stage": "feature",
        "status": "ok",
        "detector": format!("{:?}", args.detector).to_lowercase(),
        "layout": format!("{layout:?}"),
        "num_captures": num_captures,
        "num_cameras": num_phys_cameras,
        "merge_multicaps": args.merge_multicaps,
        "num_images": per_image.len(),
        "total_features": total_features,
        "per_image": per_image,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    std::fs::write(
        stage_dir.join("report.json"),
        serde_json::to_string_pretty(&payload)?,
    )?;
    let log_path = project.record_log("feature", &payload)?;
    println!(
        "Detected {total_features} features across {} images ({:?}). Report: {}",
        per_image.len(),
        args.detector,
        stage_dir.join("report.json").display()
    );
    let _ = log_path;
    Ok(())
}

/// Concatenates each camera's feature sets across captures into one feature
/// set per `(camera, image slot)`.
///
/// This is the fixed-rig case: the cameras never moved, only the scene did, so
/// every capture's observations are valid from the *same* viewpoint and belong
/// to the same pose. Merging turns N captures into N times the observations of
/// one image rather than N images that share no features - which is what makes
/// the rig reconstructable when each individual capture shows too few markers
/// to constrain anything. Corner descriptors keep their `capture_id`, so
/// markers that moved between captures stay distinct 3D points inside the
/// merged set.
fn merge_across_captures(
    records: Vec<(dataset::DiscoveredImage, u32, u32, sfm_core::FeatureSet)>,
    num_captures: usize,
) -> Result<Vec<(dataset::DiscoveredImage, u32, u32, sfm_core::FeatureSet)>> {
    if num_captures < 2 {
        println!("--merge-multicaps: only one capture present, nothing to merge");
        return Ok(records);
    }
    // Grouped by (camera, slot) so a camera holding several shots per capture
    // merges shot-for-shot rather than pooling them.
    let mut groups: std::collections::BTreeMap<(u32, u32), Vec<usize>> = Default::default();
    for (i, (d, ..)) in records.iter().enumerate() {
        groups.entry((d.camera_id, d.image_index)).or_default().push(i);
    }

    let mut out = Vec::with_capacity(groups.len());
    for ((camera_id, image_index), members) in groups {
        let (w, h) = (records[members[0]].1, records[members[0]].2);
        for &m in &members {
            if (records[m].1, records[m].2) != (w, h) {
                bail!(
                    "--merge-multicaps: camera {camera_id} slot {image_index} mixes image sizes \
                     ({}x{} vs {}x{}); merged images must come from the same unmoved camera",
                    w, h, records[m].1, records[m].2
                );
            }
        }
        let mut keypoints = Vec::new();
        let mut marker_data: Vec<u8> = Vec::new();
        let mut float_data: Vec<f32> = Vec::new();
        let mut float_dim = 0u32;
        let mut binary_data: Vec<u8> = Vec::new();
        let mut binary_stride = 0u32;
        for &m in &members {
            let fs = &records[m].3;
            keypoints.extend_from_slice(&fs.keypoints);
            match &fs.descriptors {
                sfm_core::Descriptors::MarkerCorner { data } => marker_data.extend_from_slice(data),
                sfm_core::Descriptors::Float32 { dim, data } => {
                    float_dim = *dim;
                    float_data.extend_from_slice(data);
                }
                sfm_core::Descriptors::Binary { bytes_per_descriptor, data } => {
                    binary_stride = *bytes_per_descriptor;
                    binary_data.extend_from_slice(data);
                }
            }
        }
        let descriptors = if !marker_data.is_empty() {
            sfm_core::Descriptors::MarkerCorner { data: marker_data }
        } else if !float_data.is_empty() {
            sfm_core::Descriptors::Float32 { dim: float_dim, data: float_data }
        } else {
            sfm_core::Descriptors::Binary {
                bytes_per_descriptor: binary_stride,
                data: binary_data,
            }
        };
        let first = &records[members[0]].0;
        // Name the merged row for the camera and slot it represents, since it
        // no longer corresponds to any single file on disk.
        let name = format!("cam{camera_id:03}/slot{image_index:03}");
        out.push((
            dataset::DiscoveredImage {
                path: first.path.clone(),
                capture_id: first.capture_id,
                camera_id,
                image_index,
                name,
            },
            w,
            h,
            sfm_core::FeatureSet { keypoints, descriptors },
        ));
    }
    println!(
        "--merge-multicaps: merged {num_captures} captures into {} per-camera feature set(s)",
        out.len()
    );
    Ok(out)
}

fn cmd_match(args: MatchArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let stage_dir = project.prepare_stage("match")?;

    if args.matcher != MatcherArg::MnnRatio {
        bail!(
            "matcher {:?} is not implemented yet (see PLAN.md); available now: mnn-ratio",
            args.matcher
        );
    }
    if !matches!(
        args.pairing,
        PairingArg::Exhaustive | PairingArg::Sequential | PairingArg::VocabTree
    ) {
        bail!(
            "pairing {:?} is not implemented yet (see PLAN.md); available now: exhaustive, sequential, vocab-tree",
            args.pairing
        );
    }
    if args.gpu {
        eprintln!("note: --gpu has no effect for mnn-ratio (classical CPU matcher); GPU is used once LightGlue is implemented");
    }

    let db = Database::open(&project.database_path())?;
    let images = db.list_images()?;
    if images.len() < 2 {
        bail!("need at least 2 images with extracted features to match; run `sfm extract` first");
    }
    let cameras: HashMap<u32, Camera> = db
        .list_cameras()?
        .into_iter()
        .map(|c| (c.camera_id, c))
        .collect();
    let features: Vec<sfm_core::FeatureSet> = images
        .iter()
        .map(|(id, ..)| db.load_features(*id))
        .collect::<Result<_>>()
        .context("loading extracted features (run `sfm extract` first)")?;

    let n = images.len();
    let pairs = match args.pairing {
        PairingArg::Exhaustive => sfm_match::exhaustive_pairs(n),
        PairingArg::Sequential => sfm_match::sequential_pairs(n, args.window as usize),
        PairingArg::VocabTree => {
            // `--window` doubles as "how many retrieval candidates per image"
            // here, so the one knob that controls pair count keeps meaning
            // the same thing across pairing strategies.
            let vparams = sfm_match::VocabParams {
                num_neighbors: args.window as usize,
                ..Default::default()
            };
            match sfm_match::vocab_tree_pairs(&features, &vparams) {
                Some(p) => {
                    eprintln!(
                        "vocab-tree retrieval: {} candidate pairs from {n} images ({:.1}% of exhaustive)",
                        p.len(),
                        100.0 * p.len() as f64 / (n * (n - 1) / 2).max(1) as f64
                    );
                    p
                }
                None => {
                    // Marker/fiducial descriptors have no metric for
                    // retrieval to work with - fall back rather than
                    // silently returning no pairs.
                    eprintln!(
                        "note: vocab-tree retrieval does not apply to these descriptors (fiducial markers); falling back to exhaustive pairing"
                    );
                    sfm_match::exhaustive_pairs(n)
                }
            }
        }
        _ => unreachable!("checked above"),
    };

    let params = sfm_match::VerificationParams::default();
    let results: Vec<Option<(usize, usize, sfm_core::TwoViewGeometryRecord)>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let cam_i = &cameras[&images[i].1].model;
            let cam_j = &cameras[&images[j].1].model;
            sfm_match::match_and_verify(&features[i], &features[j], cam_i, cam_j, &params)
                .map(|rec| (i, j, rec))
        })
        .collect();

    db.clear_geometries()?;
    let mut num_verified = 0usize;
    let mut total_inliers = 0usize;
    let mut per_pair = Vec::new();
    for (i, j, rec) in results.into_iter().flatten() {
        let (id1, id2) = (images[i].0, images[j].0);
        db.store_geometry(id1, id2, &rec)?;
        num_verified += 1;
        total_inliers += rec.inlier_matches.len();
        per_pair.push(serde_json::json!({
            "image_id1": id1,
            "image_id2": id2,
            "num_inliers": rec.inlier_matches.len(),
        }));
    }

    let payload = serde_json::json!({
        "stage": "match",
        "status": "ok",
        "pairing": format!("{:?}", args.pairing).to_lowercase(),
        "matcher": "mnn-ratio",
        "num_pairs_attempted": pairs.len(),
        "num_pairs_verified": num_verified,
        "total_inlier_matches": total_inliers,
        "pairs": per_pair,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    std::fs::write(
        stage_dir.join("report.json"),
        serde_json::to_string_pretty(&payload)?,
    )?;
    let log_path = project.record_log("match", &payload)?;
    println!(
        "Verified {num_verified}/{} pairs, {total_inliers} inlier matches total. Logged to {}.",
        pairs.len(),
        log_path.display()
    );
    Ok(())
}

fn cmd_map(args: MapArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let stage_dir = project.prepare_stage("map")?;

    let db = Database::open(&project.database_path())?;
    let images = db.list_images()?;
    if images.len() < 2 {
        bail!("need at least 2 images with extracted features to map; run `sfm extract` first");
    }
    let compact_of_id: HashMap<u32, usize> = images
        .iter()
        .enumerate()
        .map(|(idx, (id, ..))| (*id, idx))
        .collect();

    let cameras: HashMap<u32, Camera> = db
        .list_cameras()?
        .into_iter()
        .map(|c| (c.camera_id, c))
        .collect();
    // Known extrinsics from the project config, keyed by image file name.
    let mut known_poses: HashMap<&str, (sfm_core::Pose, bool)> = HashMap::new();
    for pc in &project.config.poses {
        let q = nalgebra::Quaternion::new(
            pc.quaternion[0],
            pc.quaternion[1],
            pc.quaternion[2],
            pc.quaternion[3],
        );
        let pose = sfm_core::Pose {
            rotation: nalgebra::UnitQuaternion::from_quaternion(q),
            translation: Vector3::new(pc.translation[0], pc.translation[1], pc.translation[2]),
        };
        known_poses.insert(pc.image.as_str(), (pose, pc.fixed.unwrap_or(false)));
    }
    let known_names: std::collections::HashSet<&str> =
        project.config.poses.iter().map(|p| p.image.as_str()).collect();
    let seen_names: std::collections::HashSet<&str> =
        images.iter().map(|(_, _, n, ..)| n.as_str()).collect();
    let mut missing: Vec<&&str> = known_names.difference(&seen_names).collect();
    if !missing.is_empty() {
        missing.sort();
        bail!(
            "{} [[poses]] entr(y/ies) name images that are not in this project: {missing:?} \
             (check the file names match exactly, including any directory prefix)",
            missing.len()
        );
    }

    let image_inputs: Vec<sfm_reconstruction::ImageInput> = images
        .iter()
        .map(|(id, camera_id, name, ..)| {
            let (initial_pose, pose_fixed) = match known_poses.get(name.as_str()) {
                Some(&(pose, fixed)) => (Some(pose), fixed),
                None => (None, false),
            };
            Ok(sfm_reconstruction::ImageInput {
                image_id: *id,
                camera_id: *camera_id,
                name: name.clone(),
                features: db.load_features(*id)?,
                initial_pose,
                pose_fixed,
            })
        })
        .collect::<Result<_>>()
        .context("loading extracted features (run `sfm extract` first)")?;

    let geometry_pairs = db.list_geometry_pairs()?;
    if geometry_pairs.is_empty() {
        bail!("no verified image pairs found; run `sfm match` first");
    }
    let pairs: Vec<sfm_reconstruction::PairInput> = geometry_pairs
        .iter()
        .map(|&(id1, id2)| {
            let geometry = db.load_geometry(id1, id2)?;
            Ok(sfm_reconstruction::PairInput {
                i: compact_of_id[&id1],
                j: compact_of_id[&id2],
                geometry,
            })
        })
        .collect::<Result<_>>()?;

    // Cameras whose intrinsics the config pins. Resolved from the declared
    // camera globs back to the ids `sfm extract` assigned, by matching the
    // same glob against the same file names.
    let mut fixed_cameras: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for cfg in &project.config.cameras {
        if cfg.refine == Some(false) {
            for (_, camera_id, name, ..) in &images {
                if project::glob_match(&cfg.images, name) {
                    fixed_cameras.insert(*camera_id);
                }
            }
        }
    }
    if !fixed_cameras.is_empty() {
        eprintln!(
            "holding intrinsics fixed for camera(s) {:?} (refine = false)",
            {
                let mut v: Vec<u32> = fixed_cameras.iter().copied().collect();
                v.sort_unstable();
                v
            }
        );
    }

    let input = sfm_reconstruction::ReconstructionInput {
        images: image_inputs,
        cameras,
        pairs,
        fixed_cameras,
    };
    let pipeline_name = match args.pipeline {
        PipelineArg::Incremental => "incremental",
        PipelineArg::Global => "global",
    };
    let recon = match args.pipeline {
        PipelineArg::Incremental => sfm_reconstruction::run_incremental(
            &input,
            &sfm_reconstruction::IncrementalParams::default(),
        ),
        PipelineArg::Global => {
            sfm_reconstruction::run_global(&input, &sfm_reconstruction::GlobalParams::default())
        }
    };

    if recon.images.is_empty() {
        bail!("reconstruction failed to register any images; check `sfm match` results (too few/weak verified pairs?)");
    }

    sfm_io::write_colmap_model(&recon, &project.sparse_dir())
        .map_err(|e| anyhow::anyhow!("writing sparse model: {e}"))?;

    let payload = serde_json::json!({
        "stage": "map",
        "status": "ok",
        "pipeline": pipeline_name,
        "num_images_input": images.len(),
        "num_images_registered": recon.images.len(),
        "num_points3d": recon.points3d.len(),
        "mean_reprojection_error_px": recon.mean_reprojection_error(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    std::fs::write(
        stage_dir.join("report.json"),
        serde_json::to_string_pretty(&payload)?,
    )?;
    let log_path = project.record_log("map", &payload)?;
    println!(
        "Registered {}/{} images, {} points3d, mean reprojection error {:.3}px. Wrote {}. Logged to {}.",
        recon.images.len(),
        images.len(),
        recon.points3d.len(),
        recon.mean_reprojection_error(),
        project.sparse_dir().display(),
        log_path.display()
    );
    Ok(())
}

/// Standalone global bundle-adjustment pass on an existing `sparse/0` model:
/// load it via `sfm-io`, convert to `sfm_ba::BaInput`, run BA, write the
/// refined poses/points/(optionally) intrinsics back to the same directory.
/// Deliberately plumbing-only, per PLAN.md - the iterative re-triangulation
/// half of "refine" (recomputing 3D positions from extended tracks, not just
/// re-optimizing existing ones) is still future work.
fn cmd_refine(args: RefineArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let sparse_dir = project.sparse_dir();
    let mut recon = sfm_io::read_colmap_model(&sparse_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to read sparse model from {} (run `sfmtory map` first): {e}",
            sparse_dir.display()
        )
    })?;

    if recon.images.len() < 2 {
        bail!(
            "need at least 2 registered images in {} to refine",
            sparse_dir.display()
        );
    }
    if recon.points3d.is_empty() {
        bail!("no 3D points in {} to refine", sparse_dir.display());
    }

    // Compact-index remap, same pattern as `sfm-reconstruction`'s in-loop BA:
    // one pose per registered image, one intrinsics block per physical
    // camera (shared across images using it), one point per `points3d` entry.
    let image_ids: Vec<u32> = recon.images.keys().copied().collect();
    let image_pos: HashMap<u32, usize> = image_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut camera_list: Vec<CameraModel> = Vec::new();
    let mut camera_id_list: Vec<u32> = Vec::new();
    let mut camera_index_of: HashMap<u32, usize> = HashMap::new();
    let mut camera_of_image: Vec<usize> = Vec::with_capacity(image_ids.len());
    for &id in &image_ids {
        let camera_id = recon.images[&id].camera_id;
        let idx = *camera_index_of.entry(camera_id).or_insert_with(|| {
            camera_list.push(recon.cameras[&camera_id].model);
            camera_id_list.push(camera_id);
            camera_list.len() - 1
        });
        camera_of_image.push(idx);
    }

    let poses: Vec<sfm_core::Pose> = image_ids.iter().map(|id| recon.images[id].pose).collect();

    let point_ids: Vec<u64> = recon.points3d.keys().copied().collect();
    let point_pos: HashMap<u64, usize> = point_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let points: Vec<Vector3<f64>> = point_ids.iter().map(|id| recon.points3d[id].xyz).collect();

    let mut observations = Vec::new();
    for (&pid, &compact_p) in &point_pos {
        for te in &recon.points3d[&pid].track {
            let (Some(&compact_img), Some(image)) =
                (image_pos.get(&te.image_id), recon.images.get(&te.image_id))
            else {
                continue;
            };
            let Some(&(x, y)) = image.keypoints.get(te.point2d_idx as usize) else {
                continue;
            };
            observations.push(sfm_ba::Observation {
                image_idx: compact_img,
                point_idx: compact_p,
                x: x as f64,
                y: y as f64,
            });
        }
    }
    if observations.is_empty() {
        bail!("no observations linking tracked points to registered images; nothing to refine");
    }

    // Reprojection error is invariant under a similarity transform of the
    // whole scene (see `BaInput::fixed_poses`'s docs), so at least one pose
    // must be anchored - image 0 (in id order) is as good a choice as any
    // for a standalone refine pass.
    let fixed_poses: Vec<bool> = (0..image_ids.len()).map(|i| i == 0).collect();
    let (fixed_cameras, fixed_camera_params) = if args.refine_intrinsics {
        (
            vec![false; camera_list.len()],
            sfm_ba::default_fixed_params_mask(&camera_list),
        )
    } else {
        (vec![true; camera_list.len()], Vec::new())
    };

    let robust_loss = match args.robust_loss {
        RobustLossArg::Huber => sfm_ba::RobustLoss::Huber,
        RobustLossArg::Cauchy => sfm_ba::RobustLoss::Cauchy,
    };
    let ba_params = sfm_ba::BaParams {
        robust_loss,
        max_iterations: 200,
        ..Default::default()
    };
    let ba_input = sfm_ba::BaInput {
        camera_of_image,
        cameras: camera_list,
        poses,
        points,
        observations,
        fixed_poses,
        fixed_cameras,
        fixed_camera_params,
    };

    let before_error = recon.mean_reprojection_error();
    let output = sfm_ba::bundle_adjust(ba_input, &ba_params);

    for (compact, &id) in image_ids.iter().enumerate() {
        recon.images.get_mut(&id).unwrap().pose = output.poses[compact];
    }
    for (compact, &id) in point_ids.iter().enumerate() {
        recon.points3d.get_mut(&id).unwrap().xyz = output.points[compact];
    }
    if args.refine_intrinsics {
        for (idx, &cam_id) in camera_id_list.iter().enumerate() {
            recon.cameras.get_mut(&cam_id).unwrap().model = output.cameras[idx];
        }
    }

    // Recompute each point's reported error (pixel-space, including
    // distortion) against the just-refined poses/intrinsics - same
    // formula `sfm-reconstruction::assemble_reconstruction` uses.
    for point in recon.points3d.values_mut() {
        let mut total = 0.0;
        let mut n = 0usize;
        for te in &point.track {
            let Some(image) = recon.images.get(&te.image_id) else {
                continue;
            };
            let Some(&(x, y)) = image.keypoints.get(te.point2d_idx as usize) else {
                continue;
            };
            let cam = &recon.cameras[&image.camera_id].model;
            let pc = image.pose.transform_point(&point.xyz);
            if pc.z > 1e-9 {
                let (px, py) = cam.project(&pc);
                total += ((px - x as f64).powi(2) + (py - y as f64).powi(2)).sqrt();
                n += 1;
            }
        }
        if n > 0 {
            point.error = total / n as f64;
        }
    }

    let after_error = recon.mean_reprojection_error();

    sfm_io::write_colmap_model(&recon, &sparse_dir)
        .map_err(|e| anyhow::anyhow!("writing refined sparse model: {e}"))?;

    let payload = serde_json::json!({
        "stage": "refine",
        "status": "ok",
        "robust_loss": format!("{:?}", args.robust_loss),
        "refine_intrinsics": args.refine_intrinsics,
        "num_images": image_ids.len(),
        "num_points3d": point_ids.len(),
        "initial_cost": output.initial_cost,
        "final_cost": output.final_cost,
        "iterations_run": output.iterations_run,
        "mean_reprojection_error_px_before": before_error,
        "mean_reprojection_error_px_after": after_error,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    let log_path = project.record_log("refine", &payload)?;
    println!(
        "Refined {} images / {} points3d: mean reprojection error {:.3}px -> {:.3}px ({} LM iterations). Wrote {}. Logged to {}.",
        image_ids.len(),
        point_ids.len(),
        before_error,
        after_error,
        output.iterations_run,
        sparse_dir.display(),
        log_path.display()
    );
    Ok(())
}

fn cmd_export(args: ExportArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let sparse_dir = project.sparse_dir();
    let recon = sfm_io::read_colmap_model(&sparse_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to read sparse model from {} (run `sfm map` first): {e}",
            sparse_dir.display()
        )
    })?;
    let out = args.out.clone().unwrap_or_else(|| project.export_dir());

    match args.format {
        ExportFormatArg::ColmapText => {
            sfm_io::write_colmap_model(&recon, &out)?;
        }
        ExportFormatArg::NerfTransforms => {
            let out_file = if out.extension().is_some() {
                out.clone()
            } else {
                std::fs::create_dir_all(&out)?;
                out.join("transforms.json")
            };
            sfm_io::write_transforms(&recon, &out_file)?;
        }
    }

    let payload = serde_json::json!({
        "stage": "export",
        "status": "ok",
        "format": format!("{:?}", args.format),
        "out": out.to_string_lossy(),
        "num_images": recon.images.len(),
        "num_points3d": recon.points3d.len(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    project.record_log("export", &payload)?;
    println!(
        "Exported {} images / {} points to {}",
        recon.images.len(),
        recon.points3d.len(),
        out.display()
    );
    Ok(())
}

/// Reprojection statistics recomputed from geometry, rather than read back
/// from whatever a previous bundle adjustment happened to store.
struct ReprojStats {
    mean: f64,
    median: f64,
    p95: f64,
    max: f64,
    num_observations: usize,
    num_points: usize,
    num_images: usize,
    /// Observations whose point falls behind the camera, which a stored
    /// per-point average silently folds away.
    num_behind_camera: usize,
}

/// Recomputes reprojection error from poses, intrinsics and 3D points.
///
/// Deliberately not `Reconstruction::mean_reprojection_error`, which averages
/// the `error` field each point carries from the last bundle adjustment that
/// touched it. That value is fine as a progress signal inside the pipeline but
/// is the wrong thing for evaluation: it reports what the optimizer believed,
/// not what the model on disk actually does, so it cannot catch a model that
/// was written out inconsistently, and it is not comparable against another
/// tool's model whose `error` column was produced by different code.
fn recompute_reprojection(recon: &sfm_core::Reconstruction) -> ReprojStats {
    let mut errors: Vec<f64> = Vec::new();
    let mut num_behind_camera = 0usize;
    for point in recon.points3d.values() {
        for t in &point.track {
            let Some(image) = recon.images.get(&t.image_id) else {
                continue;
            };
            let Some(camera) = recon.cameras.get(&image.camera_id) else {
                continue;
            };
            let Some(&(u, v)) = image.keypoints.get(t.point2d_idx as usize) else {
                continue;
            };
            let pc = image.pose.transform_point(&point.xyz);
            if pc.z <= 1e-9 {
                num_behind_camera += 1;
                continue;
            }
            let (px, py) = camera.model.project(&pc);
            errors.push(((px - u as f64).powi(2) + (py - v as f64).powi(2)).sqrt());
        }
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| -> f64 {
        if errors.is_empty() {
            0.0
        } else {
            errors[((errors.len() - 1) as f64 * q).round() as usize]
        }
    };
    ReprojStats {
        mean: if errors.is_empty() {
            0.0
        } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        },
        median: pick(0.5),
        p95: pick(0.95),
        max: errors.last().copied().unwrap_or(0.0),
        num_observations: errors.len(),
        num_points: recon.points3d.len(),
        num_images: recon.images.len(),
        num_behind_camera,
    }
}

fn print_reproj(label: &str, s: &ReprojStats) {
    println!(
        "  {label:<10} images {:>4}  points {:>7}  obs {:>8}  mean {:.4}px  median {:.4}px  p95 {:.4}px  max {:.3}px",
        s.num_images, s.num_points, s.num_observations, s.mean, s.median, s.p95, s.max
    );
    if s.num_behind_camera > 0 {
        println!(
            "  {:<10} {} observation(s) fall behind the camera and were excluded",
            "", s.num_behind_camera
        );
    }
}

/// Parses `--gt-focal`: a bare number, or a file holding either a bare number
/// or a 3x3 K matrix whose first entry is fx.
fn parse_gt_focal(spec: &str) -> Result<f64> {
    if let Ok(v) = spec.trim().parse::<f64>() {
        return Ok(v);
    }
    let text = std::fs::read_to_string(spec)
        .with_context(|| format!("--gt-focal: {spec} is neither a number nor a readable file"))?;
    let first = text
        .split_whitespace()
        .next()
        .context("--gt-focal file is empty")?;
    first
        .parse::<f64>()
        .with_context(|| format!("--gt-focal: could not read a focal length from {spec}"))
}

fn focal_summary(recon: &sfm_core::Reconstruction) -> Vec<(u32, f64)> {
    let mut v: Vec<(u32, f64)> = recon
        .cameras
        .iter()
        .map(|(id, c)| {
            let (fx, fy) = c.model.focal_lengths();
            (*id, (fx + fy) / 2.0)
        })
        .collect();
    v.sort_by_key(|(id, _)| *id);
    v
}

fn cmd_eval(args: EvalArgs) -> Result<()> {
    let ours_dir = match &args.ours {
        Some(d) => d.clone(),
        None => Project::open(&args.project)?.sparse_dir(),
    };
    let recon = sfm_io::read_colmap_model(&ours_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to read model from {} (run `sfmtory map` first): {e}",
            ours_dir.display()
        )
    })?;

    println!("Model: {}", ours_dir.display());
    let ours = recompute_reprojection(&recon);
    println!("Reprojection error (recomputed from geometry):");
    print_reproj("ours", &ours);

    let baseline_stats = args
        .baseline
        .as_ref()
        .map(|b| -> Result<_> {
            let m = sfm_io::read_colmap_model(b)
                .map_err(|e| anyhow::anyhow!("reading baseline {}: {e}", b.display()))?;
            let st = recompute_reprojection(&m);
            print_reproj("baseline", &st);
            Ok((m, st))
        })
        .transpose()?;

    // Focal lengths, and the error against whichever reference was given.
    println!("Focal lengths:");
    let ours_focals = focal_summary(&recon);
    let reference: Option<(String, f64)> = if let Some(spec) = &args.gt_focal {
        Some((format!("--gt-focal {spec}"), parse_gt_focal(spec)?))
    } else if let Some(gt_dir) = &args.gt {
        let m = sfm_io::read_colmap_model(gt_dir)
            .map_err(|e| anyhow::anyhow!("reading ground truth {}: {e}", gt_dir.display()))?;
        let f = focal_summary(&m);
        if f.is_empty() {
            bail!("ground-truth model {} has no cameras", gt_dir.display());
        }
        // A single mean is the honest summary when the reference has several
        // cameras: matching them up to ours by id would assume an ordering
        // neither model promises.
        let mean = f.iter().map(|(_, v)| *v).sum::<f64>() / f.len() as f64;
        Some((format!("{}", gt_dir.display()), mean))
    } else {
        None
    };

    let mut focal_errors = Vec::new();
    for (id, f) in &ours_focals {
        match &reference {
            Some((_, gt)) => {
                let err = (f - gt).abs() / gt * 100.0;
                focal_errors.push(err);
                println!("  camera {id:<4} f = {f:>10.3} px   error vs reference {err:>6.3}%");
            }
            None => println!("  camera {id:<4} f = {f:>10.3} px"),
        }
    }
    if let Some((src, gt)) = &reference {
        println!("  reference focal {gt:.3} px (from {src})");
        if focal_errors.len() > 1 {
            let mean = focal_errors.iter().sum::<f64>() / focal_errors.len() as f64;
            let worst = focal_errors.iter().cloned().fold(0.0f64, f64::max);
            println!("  mean focal error {mean:.3}%, worst {worst:.3}%");
        }
    }
    if let Some((base, _)) = &baseline_stats {
        println!("Baseline focal lengths:");
        for (id, f) in focal_summary(base) {
            println!("  camera {id:<4} f = {f:>10.3} px");
        }
    }

    if args.baseline.is_some() {
        eprintln!(
            "note: pose-accuracy comparison (Umeyama-aligned camera centres) is not implemented; \
             the numbers above compare reprojection and calibration only"
        );
    }
    Ok(())
}

fn cmd_debug_pair(args: DebugPairArgs) -> Result<()> {
    let project = Project::open(&args.project)?;
    let db = Database::open(&project.database_path())?;
    let images = db.list_images()?;
    let cameras: HashMap<u32, Camera> = db
        .list_cameras()?
        .into_iter()
        .map(|c| (c.camera_id, c))
        .collect();
    let cam_id_of = |id: u32| {
        images
            .iter()
            .find(|(iid, ..)| *iid == id)
            .map(|(_, cid, ..)| *cid)
            .unwrap()
    };

    let f1 = db.load_features(args.image1)?;
    let f2 = db.load_features(args.image2)?;
    println!("image {}: {} features", args.image1, f1.len());
    println!("image {}: {} features", args.image2, f2.len());

    for ratio in [0.6, 0.7, 0.8, 0.9, 1.0] {
        let putative = sfm_match::match_descriptors(
            &f1.descriptors,
            &f2.descriptors,
            &sfm_match::MatchParams {
                ratio_threshold: ratio,
            },
        );
        println!("ratio={ratio}: {} putative matches", putative.len());
    }

    let putative = sfm_match::match_descriptors(
        &f1.descriptors,
        &f2.descriptors,
        &sfm_match::MatchParams::default(),
    );
    let pts1: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(i, _)| {
            let kp = f1.keypoints[i as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();
    let pts2: Vec<(f64, f64)> = putative
        .iter()
        .map(|&(_, j)| {
            let kp = f2.keypoints[j as usize];
            (kp.x as f64, kp.y as f64)
        })
        .collect();
    let cam1 = &cameras[&cam_id_of(args.image1)].model;
    let cam2 = &cameras[&cam_id_of(args.image2)].model;

    for threshold_px in [2.0, 4.0, 8.0, 16.0, 32.0] {
        match sfm_geometry::estimate_two_view_geometry(&pts1, &pts2, cam1, cam2, threshold_px, 5000) {
            Some(geom) => {
                let ratio = geom.num_inliers as f64 / putative.len().max(1) as f64;
                println!("threshold={threshold_px}px: {} / {} inliers (ratio {:.2})", geom.num_inliers, putative.len(), ratio);
            }
            None => println!("threshold={threshold_px}px: verification failed entirely (< 8 putative or RANSAC found nothing)"),
        }
    }
    Ok(())
}

fn cmd_run(args: RunArgs) -> Result<()> {
    println!("`sfmtory run` chains feature -> match -> map -> export; running each stage now.");
    cmd_feature(FeatureArgs {
        project: args.project.clone(),
        detector: args.detector,
        aruco_dict: None,
        max_features: None,
        find_params: false,
        merge_multicaps: false,
        gpu: false,
    })?;
    cmd_match(MatchArgs {
        project: args.project.clone(),
        pairing: args.pairing,
        matcher: args.matcher,
        window: 10,
        gpu: false,
    })?;
    cmd_map(MapArgs {
        project: args.project.clone(),
        pipeline: args.pipeline,
    })?;
    cmd_export(ExportArgs {
        project: args.project.clone(),
        format: ExportFormatArg::ColmapText,
        out: None,
    })?;
    Ok(())
}
