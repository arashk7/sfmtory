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
mod modelselect;
mod progress;
mod project;
mod rebuild;
mod rig;

use std::collections::{BTreeMap, BTreeSet, HashMap};
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
    /// Calibrate a rig of fixed cameras from several captures of a moved target.
    Rig(RigArgs),
    /// Choose a camera model per camera by held-out reprojection error.
    #[command(name = "select-model")]
    SelectModel(SelectModelArgs),
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
    /// pipeline reads, from the raw tree and the layout in sfm.toml.
    Link(DatasetLinkArgs),
    /// List the layouts this project declares.
    Layouts(DatasetProjectArgs),
    /// Add a named layout to sfm.toml.
    AddLayout(DatasetAddLayoutArgs),
    /// Remove a named layout from sfm.toml.
    RemoveLayout(DatasetRemoveLayoutArgs),
}

#[derive(clap::Args)]
struct DatasetProjectArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
}

#[derive(clap::Args)]
struct DatasetAddLayoutArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Name to refer to this layout by.
    #[arg(long)]
    name: String,
    /// One role per path level, outermost first, ending with the file.
    /// Roles: capture, camera, image (or frame), ignore. A level may instead
    /// be a pattern such as `cam{camera}_{image}` when one name carries two
    /// ids.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    layers: Vec<String>,
    /// Raw image tree, relative to the project root. Defaults to images_dir.
    #[arg(long)]
    source: Option<PathBuf>,
    /// Use this layout when `dataset link` is not given a `--layout`.
    #[arg(long)]
    default: bool,
}

#[derive(clap::Args)]
struct DatasetRemoveLayoutArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Name of the layout to remove.
    name: String,
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
    /// Which named layout to build from. Optional when the project has one
    /// layout, or one marked `default = true`.
    #[arg(long)]
    layout: Option<String>,
}

#[derive(clap::Args)]
struct RigArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Cameras a capture must share with the reference before it can be
    /// aligned to it.
    #[arg(long, default_value_t = 3)]
    min_shared: usize,
    /// Write the averaged rig to `cache/rig/sparse/0`.
    #[arg(long)]
    write: bool,
}

#[derive(clap::Args)]
struct SelectModelArgs {
    /// Project directory. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Write the winning model into sfm.toml as a `[[cameras]]` entry.
    #[arg(long)]
    apply: bool,
    /// Score folds by image even when capture information is available.
    #[arg(long)]
    fold_by_image: bool,
    /// Rebuild the reconstruction once per candidate model instead of scoring
    /// candidates against the existing structure.
    ///
    /// Slower - minutes rather than seconds - and the only mode that can detect
    /// a wrong *projection family*, such as a fisheye lens being modelled as a
    /// rectilinear one. Scoring against existing structure cannot: those points
    /// were triangulated by the model being judged, so it always fits them.
    #[arg(long)]
    rebuild: bool,
}

#[cfg(feature = "gui")]
#[derive(clap::Args)]
struct GuiArgs {
    /// Project directory to open. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Which view to open on. Defaults to the guided start screen; pass
    /// `--view graph` or similar to go straight to the diagnostic that prompted
    /// opening the viewer at all.
    #[arg(long, value_enum, default_value_t = GuiViewArg::Start)]
    view: GuiViewArg,
}

#[cfg(feature = "gui")]
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum GuiViewArg {
    /// What to do next, and whether the answer will mean anything.
    Start,
    /// The 3D point cloud and camera frusta.
    Scene,
    /// The selected image with its features and residuals drawn on it.
    Image,
    /// Images as nodes, verified pairs as edges, components coloured.
    Graph,
    /// Board-orientation coverage for a planar capture.
    Coverage,
    /// Try the ArUco detector and its parameters on one frame.
    Aruco,
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
    /// Camera model for images not covered by a `[[cameras]]` entry.
    ///
    /// SIMPLE_RADIAL (one radial term) is a safe default for a narrow lens and
    /// too simple for a wide one. See the Camera setup section of the README.
    #[arg(long, default_value = "SIMPLE_RADIAL", value_parser = CAMERA_MODELS)]
    camera_model: String,
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
    /// Every pair *within* a capture, none across them.
    ///
    /// For a rig photographing a target that moves between captures, a
    /// cross-capture pair can never match: marker identities are stamped with
    /// their capture precisely so a moved marker cannot match itself. Trying
    /// them anyway is the bulk of the work in an exhaustive run - on a
    /// 5-capture, 193-camera project it is 465k pairs against 93k, five times
    /// the matching for pairs that are all guaranteed to fail.
    WithinCapture,
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
    /// Camera model for images not covered by a `[[cameras]]` entry.
    #[arg(long, default_value = "SIMPLE_RADIAL", value_parser = CAMERA_MODELS)]
    camera_model: String,
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
            DatasetAction::Layouts(args) => cmd_dataset_layouts(args),
            DatasetAction::AddLayout(args) => cmd_dataset_add_layout(args),
            DatasetAction::RemoveLayout(args) => cmd_dataset_remove_layout(args),
        },
        Commands::Rig(args) => cmd_rig(args),
        Commands::SelectModel(args) => cmd_select_model(args),
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
/// Every model `CameraModel::from_name_and_params` understands, in increasing
/// order of parameter count.
pub const CAMERA_MODELS: [&str; 7] = [
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "RADIAL3",
    "OPENCV",
    "OPENCV_FISHEYE",
];

fn default_camera_model(name: &str, w: u32, h: u32) -> Option<CameraModel> {
    let f = w.max(h) as f64 * 1.2;
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let params: Vec<f64> = match name {
        "SIMPLE_PINHOLE" => vec![f, cx, cy],
        "PINHOLE" => vec![f, f, cx, cy],
        "SIMPLE_RADIAL" => vec![f, cx, cy, 0.0],
        "RADIAL" => vec![f, cx, cy, 0.0, 0.0],
        "RADIAL3" => vec![f, cx, cy, 0.0, 0.0, 0.0],
        "OPENCV" => vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        "OPENCV_FISHEYE" => vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        _ => return None,
    };
    CameraModel::from_name_and_params(name, &params)
}

/// Each observation's capture, recovered from the fiducial descriptors.
///
/// The reconstruction does not carry capture identity - `--merge-multicaps`
/// deliberately pools several captures into one image row, and marks it
/// `capture_id = -1` precisely because no single value is right. The identity
/// survives per *observation* though: every marker corner was stamped with its
/// capture at detection time so a moved marker cannot match itself across
/// captures, and that stamp is still in the descriptor blob.
fn capture_of_observations(
    problem: &modelselect::Problem,
    db: &Database,
) -> Option<modelselect::Folds> {
    const STRIDE: usize = 12;
    let mut per_image: Vec<Vec<u32>> = Vec::with_capacity(problem.image_ids.len());
    for &id in &problem.image_ids {
        let fs = db.load_features(id).ok()?;
        let sfm_core::Descriptors::MarkerCorner { data } = &fs.descriptors else {
            return None;
        };
        per_image.push(
            (0..fs.keypoints.len())
                .map(|i| {
                    let r = data.get(i * STRIDE..i * STRIDE + 4)?;
                    Some(u32::from_le_bytes([r[0], r[1], r[2], r[3]]))
                })
                .collect::<Option<Vec<u32>>>()?,
        );
    }

    // Observations carry a point index, not a keypoint index, so recover the
    // capture by matching the observed pixel back to its keypoint.
    let mut of_observation = Vec::with_capacity(problem.observations.len());
    let mut distinct: std::collections::BTreeSet<u32> = Default::default();
    for o in &problem.observations {
        let caps = &per_image[o.image_idx];
        let fs = db.load_features(problem.image_ids[o.image_idx]).ok()?;
        let idx = fs
            .keypoints
            .iter()
            .position(|k| (k.x as f64 - o.x).abs() < 1e-3 && (k.y as f64 - o.y).abs() < 1e-3)?;
        let c = *caps.get(idx)?;
        distinct.insert(c);
        of_observation.push(c);
    }
    if distinct.len() < 2 {
        return None;
    }
    let index: HashMap<u32, usize> = distinct.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    Some(modelselect::Folds {
        of_observation: of_observation.iter().map(|c| index[c]).collect(),
        count: distinct.len(),
        kind: "capture",
    })
}

fn cmd_rig(args: RigArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let db = Database::open(&project.database_path())?;
    let all = db.list_images_with_capture()?;
    if all.is_empty() {
        bail!("no images in the project database; run `sfmtory feature` first");
    }
    if all.iter().any(|(_, _, _, cap)| *cap < 0) {
        bail!(
            "this project was built with --merge-multicaps, which pools every capture into one \n\
             image per camera and leaves nothing to compare between captures.\n\n\
             Re-run `sfmtory feature --detector aruco` without --merge-multicaps, then \n\
             `sfmtory match`, then this command. Merging helps intrinsics and costs the rig \n\
             geometry: measured on a 4-camera wall rig, merged put the centres 16.2% of their \n\
             mean spacing off a common plane, against 1.1% per capture."
        );
    }

    let mut by_capture: BTreeMap<i64, Vec<(u32, u32, String)>> = BTreeMap::new();
    for (id, camera_id, name, capture) in &all {
        by_capture
            .entry(*capture)
            .or_default()
            .push((*id, *camera_id, name.clone()));
    }
    println!(
        "{} images across {} capture(s); reconstructing each on its own",
        all.len(),
        by_capture.len()
    );
    if by_capture.len() < 2 {
        bail!(
            "a rig needs at least two captures to have anything to cross-check; this project \
             has one"
        );
    }

    let cameras: HashMap<u32, Camera> = db
        .list_cameras()?
        .into_iter()
        .map(|c| (c.camera_id, c))
        .collect();
    let geometry_pairs = db.list_geometry_pairs()?;
    if geometry_pairs.is_empty() {
        bail!("no verified image pairs; run `sfmtory match` first");
    }

    let reporter = progress::Progress::new("rig", by_capture.len());
    let mut rigs = Vec::new();
    for (capture, members) in &by_capture {
        // Each capture is its own scene: only its own images, only pairs
        // within it. Marker identities are stamped per capture anyway, so
        // there are no cross-capture pairs to lose.
        let ids: BTreeSet<u32> = members.iter().map(|(id, ..)| *id).collect();
        let compact: HashMap<u32, usize> = members
            .iter()
            .enumerate()
            .map(|(i, (id, ..))| (*id, i))
            .collect();
        let image_inputs: Vec<sfm_reconstruction::ImageInput> = members
            .iter()
            .map(|(id, camera_id, name)| {
                Ok(sfm_reconstruction::ImageInput {
                    image_id: *id,
                    camera_id: *camera_id,
                    name: name.clone(),
                    features: db.load_features(*id)?,
                    initial_pose: None,
                    pose_fixed: false,
                })
            })
            .collect::<Result<_>>()?;
        let pairs: Vec<sfm_reconstruction::PairInput> = geometry_pairs
            .iter()
            .filter(|(a, b)| ids.contains(a) && ids.contains(b))
            .map(|&(a, b)| {
                Ok(sfm_reconstruction::PairInput {
                    i: compact[&a],
                    j: compact[&b],
                    geometry: db.load_geometry(a, b)?,
                })
            })
            .collect::<Result<_>>()?;
        if pairs.is_empty() {
            reporter.tick();
            continue;
        }
        let input = sfm_reconstruction::ReconstructionInput {
            images: image_inputs,
            cameras: cameras.clone(),
            pairs,
            fixed_cameras: Default::default(),
        };
        let recon = sfm_reconstruction::run_incremental(
            &input,
            &sfm_reconstruction::IncrementalParams::default(),
        );
        if !recon.images.is_empty() {
            rigs.push(rig::CaptureRig::from_reconstruction(*capture, &recon));
        }
        reporter.tick();
    }
    eprintln!(
        "rig: per-capture reconstruction finished in {}",
        progress::human_secs(reporter.elapsed_secs())
    );

    println!("\nper capture:");
    for r in &rigs {
        println!(
            "  capture {:>4}: {:>3} camera(s), {:>6} points, {:.3}px",
            r.capture_id,
            r.centres.len(),
            r.num_points,
            r.mean_reprojection_px
        );
    }

    let solution = rig::solve(&rigs, args.min_shared)?;
    for (capture, why) in &solution.skipped {
        println!("  capture {capture} not aligned: {why}");
    }
    if !solution.agreement.is_empty() {
        println!("\nhow much of the rig each capture agrees with the reference on:");
        for (capture, inliers, shared) in &solution.agreement {
            println!(
                "  capture {capture:>4}: {inliers:>4}/{shared:<4} cameras ({:.0}%)",
                100.0 * *inliers as f64 / (*shared).max(1) as f64
            );
        }
    }

    println!(
        "\naligned to capture {} - {} camera(s), mean spacing {:.4}",
        solution.reference,
        solution.cameras.len(),
        solution.mean_spacing
    );
    println!("agreement between captures, per camera:");
    let mut worst: Vec<&rig::CameraSpread> = solution.cameras.iter().collect();
    worst.sort_by(|a, b| {
        b.rms
            .partial_cmp(&a.rms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for c in worst.iter().take(8) {
        println!(
            "  camera {:>4}: seen in {} capture(s), position spread {:.5} ({:.2}% of spacing), \
             axis spread {:.2} deg",
            c.camera_id,
            c.observations,
            c.rms,
            100.0 * c.rms / solution.mean_spacing,
            c.axis_spread_deg
        );
    }
    if solution.cameras.len() > 8 {
        println!("  ... and {} more", solution.cameras.len() - 8);
    }
    let mean_rms: f64 =
        solution.cameras.iter().map(|c| c.rms).sum::<f64>() / solution.cameras.len() as f64;
    println!(
        "  mean spread: {:.5} ({:.2}% of camera spacing)",
        mean_rms,
        100.0 * mean_rms / solution.mean_spacing
    );

    // The spread is only over captures that agreed about a camera, so on its
    // own it is survivorship bias: a camera two captures agree on looks as
    // good as one all five agree on, and it is far less determined. Say how
    // many captures actually back each number.
    let mut seen: Vec<usize> = solution.cameras.iter().map(|c| c.observations).collect();
    seen.sort_unstable();
    let median_seen = seen.get(seen.len() / 2).copied().unwrap_or(0);
    let well_backed = seen.iter().filter(|n| **n >= 3).count();
    println!(
        "  backed by: median {median_seen} of {} captures; {well_backed}/{} cameras agreed \
         across 3 or more",
        rigs.len(),
        solution.cameras.len()
    );
    if median_seen * 2 < rigs.len() {
        println!(
            "{}",
            format_args!(
                "  NOTE: most cameras agreed in fewer than half the captures, so the spread \
                 above describes the subset that agreed rather than the rig as a whole. \
                 Treat it as a lower bound on the real uncertainty."
            )
        );
    }

    // A fact about the hardware that nothing in the reconstruction was told,
    // which makes it the cheapest honest check that the result is right.
    let centres: Vec<Vector3<f64>> = solution.cameras.iter().map(|c| c.mean).collect();
    if let Some((off_plane, ratio)) = rig::planarity(&centres) {
        println!(
            "\nrig shape: cameras sit {:.2}% of their mean spacing off their own best-fit plane \
             (out-of-plane / in-plane extent {:.4})",
            100.0 * off_plane,
            ratio
        );
        println!(
            "  {}",
            if off_plane < 0.03 {
                "consistent with all cameras on one flat surface"
            } else if off_plane < 0.12 {
                "roughly planar - some curvature or noise"
            } else {
                "not planar; expected for cameras spread over several walls"
            }
        );
    }
    println!(
        "elapsed {}",
        progress::human_secs(started.elapsed().as_secs_f64())
    );

    project.record_log(
        "rig",
        &serde_json::json!({
            "stage": "rig",
            "status": "ok",
            "num_captures": by_capture.len(),
            "aligned": rigs.len() - solution.skipped.len(),
            "num_cameras": solution.cameras.len(),
            "mean_spread_fraction": mean_rms / solution.mean_spacing,
            "planarity": rig::planarity(&centres).map(|(o, _)| o),
            "elapsed_ms": started.elapsed().as_millis(),
        }),
    )?;
    let _ = args.write;
    Ok(())
}

/// Model selection by rebuilding - see `crate::rebuild` for why this exists
/// alongside the structure-scoring path.
fn cmd_select_model_rebuild(args: &SelectModelArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let MapInput { input, num_images } = load_map_input(&project)?;
    let params = sfm_reconstruction::IncrementalParams::default();

    let counts = {
        let mut n: std::collections::BTreeMap<u32, usize> = Default::default();
        for im in &input.images {
            *n.entry(im.camera_id).or_insert(0) += 1;
        }
        n
    };
    println!(
        "Rebuilding to compare camera models: {} image(s), {} camera(s).",
        num_images,
        input.cameras.len()
    );
    let few: Vec<u32> = counts
        .iter()
        .filter(|(_, n)| **n < sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS)
        .map(|(id, _)| *id)
        .collect();
    if !few.is_empty() {
        println!(
            "  {} camera(s) have fewer than {} images, so bundle adjustment cannot move",
            few.len(),
            sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
        );
        println!("  their focal; the search sweeps it instead. Each family needs its own focal -");
        println!("  a fisheye's is roughly a third of a rectilinear lens's for the same view -");
        println!("  so without this the comparison would be decided by the focal, not the model.");
    }
    println!();

    // Batched rather than per-item progress: the trials inside a pass run in
    // parallel, so there is no meaningful ordering to tick through - only the
    // completion of each pass.
    let outcome = rebuild::search(&input, &params, |done, total| {
        println!(
            "  rebuilt {done}/{total} candidates ({} elapsed)",
            progress::human_secs(started.elapsed().as_secs_f64())
        );
    });

    println!();
    if !outcome.baseline.mean_error.is_finite() {
        println!("The reconstruction as currently configured does not build, so there is no");
        println!("baseline to compare against. Fix `map` first.");
    } else {
        println!(
            "As configured now:  {} image(s), {} points, {:.3}px",
            outcome.baseline.registered, outcome.baseline.points, outcome.baseline.mean_error
        );
    }
    if !outcome.best.mean_error.is_finite() {
        bail!("no candidate model produced a usable reconstruction");
    }
    println!(
        "Best found:         {} image(s), {} points, {:.3}px",
        outcome.best.registered, outcome.best.points, outcome.best.mean_error
    );

    // Show the per-camera sweep, since the whole point is that the user can see
    // *why* a family was chosen rather than being handed a verdict.
    println!();
    println!("Per-camera search (each row rebuilt the whole reconstruction):");
    let mut by_camera: std::collections::BTreeMap<u32, Vec<&rebuild::Tried>> = Default::default();
    for t in outcome.tried.iter().filter(|t| t.camera_id.is_some()) {
        by_camera.entry(t.camera_id.unwrap()).or_default().push(t);
    }
    for (id, rows) in &by_camera {
        println!("  camera {id}:");
        // Best focal per family, so the table stays readable when the focal was
        // swept: the family is the question, the focal is a nuisance parameter.
        let mut best_per_family: std::collections::BTreeMap<&str, &rebuild::Tried> =
            Default::default();
        for t in rows {
            let e = best_per_family.entry(t.choice.model).or_insert(t);
            if t.score.mean_error < e.score.mean_error {
                *e = t;
            }
        }
        for name in rebuild::FAMILIES {
            let Some(t) = best_per_family.get(name) else {
                continue;
            };
            let chosen = outcome
                .assignment
                .get(id)
                .map(|c| c.model == t.choice.model)
                .unwrap_or(false);
            let mark = if chosen { "->" } else { "  " };
            if t.score.mean_error.is_finite() {
                println!(
                    "   {mark} {:<16} f = {:>8.1}  {:>3} img  {:>5} pts  {:>7.3} px",
                    name, t.choice.focal, t.score.registered, t.score.points, t.score.mean_error
                );
            } else {
                println!("   {mark} {name:<16} did not reconstruct");
            }
        }
    }

    println!();
    println!("Recommended:");
    for (id, c) in &outcome.assignment {
        println!(
            "  camera {id}: {} (f = {:.1}{})",
            c.model,
            c.focal,
            if c.focal_swept { ", swept" } else { "" }
        );
    }
    println!();
    println!(
        "A more complex family had to beat a simpler one by {:.0}% to be preferred; \
         reprojection",
        rebuild::PARSIMONY_MARGIN * 100.0
    );
    println!("error always falls with parameter count, so lowest-error-wins would pick the most");
    println!("complex model offered every time.");

    let globs: std::collections::BTreeMap<u32, String> = project
        .config
        .cameras
        .iter()
        .filter_map(|c| {
            let id = input
                .images
                .iter()
                .find_map(|im| project::glob_match(&c.images, &im.name).then_some(im.camera_id))?;
            Some((id, c.images.clone()))
        })
        .collect();
    let toml = rebuild::as_camera_toml(&outcome.assignment, &globs, true);

    if args.apply {
        let cfg_path = Project::config_path(&project.root);
        let existing = std::fs::read_to_string(&cfg_path).unwrap_or_default();
        if existing.contains("[[cameras]]") {
            println!();
            println!(
                "{} already declares [[cameras]]; not overwriting. Replace those blocks with:",
                cfg_path.display()
            );
            println!();
            print!("{toml}");
        } else {
            std::fs::write(&cfg_path, format!("{existing}\n{toml}"))?;
            println!();
            println!("Applied to {}. Re-run `sfmtory map`.", cfg_path.display());
        }
    } else {
        println!();
        println!("Add to sfm.toml (or re-run with --apply):");
        println!();
        print!("{toml}");
    }

    let payload = serde_json::json!({
        "stage": "select-model",
        "status": "ok",
        "mode": "rebuild",
        "num_rebuilds": outcome.tried.len() + 1,
        "baseline_error_px": outcome.baseline.mean_error,
        "best_error_px": outcome.best.mean_error,
        "recommended": outcome.assignment.iter()
            .map(|(id, c)| (id.to_string(), c.model))
            .collect::<std::collections::BTreeMap<_, _>>(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    project.record_log("select-model", &payload)?;
    Ok(())
}

fn cmd_select_model(args: SelectModelArgs) -> Result<()> {
    if args.rebuild {
        return cmd_select_model_rebuild(&args);
    }
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let dir = project.sparse_dir();
    let recon = sfm_io::read_colmap_model(&dir)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", dir.display()))?;
    let problem = modelselect::Problem::from_reconstruction(&recon)?;

    // Whole captures are the right folds when they exist; whole images
    // otherwise. Never a random split - see the module docs.
    let folds = if args.fold_by_image {
        None
    } else {
        Database::open(&project.database_path())
            .ok()
            .and_then(|db| capture_of_observations(&problem, &db))
    }
    .unwrap_or_else(|| modelselect::Folds {
        of_observation: problem.observations.iter().map(|o| o.image_idx).collect(),
        count: problem.poses.len(),
        kind: "image",
    });

    println!(
        "{} camera(s), {} observations, {} folds by {}",
        problem.cameras.len(),
        problem.observations.len(),
        folds.count,
        folds.kind
    );
    if folds.count < 2 {
        bail!("need at least 2 folds to score a model on data it was not fitted to");
    }

    // Refuse to rank models the data cannot separate. See `fold_overlap`.
    let overlap = modelselect::fold_overlap(&problem, &folds);
    println!(
        "{:.0}% of points are observed in more than one fold",
        overlap.shared_fraction * 100.0
    );
    if overlap.shared_fraction < modelselect::MIN_SHARED_POINTS {
        println!();
        bail!(
            "these folds share almost no points ({:.0}% of {}), so held-out reprojection error \
             cannot tell the models apart - a point that only its own fold observes is scored \
             against structure that fold helped place.\n\n\
             With fiducial folds this is structural: every marker is stamped with its capture, \
             so each point belongs to exactly one capture by construction.\n\n\
             What does work here is comparing whole reconstructions: run `map` with each \
             candidate set via `[[cameras]] model = ...` and compare point count and mean \
             reprojection error. On a 4-camera rig that separated the true models clearly \
             (0.921px/677pts) from a swapped assignment (1.239px/649pts) while this comparison \
             saw nothing.\n\n\
             Pass --fold-by-image to score anyway, understanding the above.",
            overlap.shared_fraction * 100.0,
            overlap.num_points
        );
    }

    // Independent of the model comparison below, and able to see something it
    // structurally cannot - see `radial_residual_trend`.
    let mut suspect = Vec::new();
    for (idx, id) in problem.camera_ids.iter().enumerate() {
        if let Some(t) = modelselect::radial_residual_trend(&problem, idx) {
            if t > modelselect::RADIAL_TREND_SUSPECT {
                suspect.push((*id, t, modelselect::trend_cause(&problem.cameras[idx])));
            }
        }
    }
    if !suspect.is_empty() {
        use modelselect::TrendCause;
        println!();
        println!(
            "{} camera(s) have residuals that grow toward the edge of the frame. Comparing",
            suspect.len()
        );
        println!("models against fixed structure cannot see this, because the points were");
        println!("triangulated by the model being judged.");
        for (id, t, _) in suspect.iter().take(8) {
            println!("   camera {id}: residual/radius correlation {t:+.2}");
        }
        // The trend says the periphery is mismodelled; only the camera's own
        // parameters say whether that is the wrong family or the right family
        // left unfitted. Recommending a wider model for a camera whose
        // coefficients are all still zero sends the user in a circle.
        let unfitted = suspect
            .iter()
            .filter(|(_, _, c)| *c == TrendCause::Unfitted)
            .count();
        if unfitted > 0 {
            println!();
            println!("{unfitted} of them still have every distortion coefficient at zero, so the");
            println!("model they have has never been fitted - a wider one would not help. See the");
            println!(
                "\"kept the focal length they started with\" warning from `map`: intrinsics are"
            );
            println!(
                "refined only for cameras with at least {} images.",
                sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
            );
        }
        if suspect.iter().any(|(_, _, c)| {
            matches!(
                c,
                TrendCause::NoDistortionTerms | TrendCause::FittedButStillCurved
            )
        }) {
            println!();
            println!("For the rest, the projection family itself is the suspect - a fisheye lens");
            println!("fitted as rectilinear looks exactly like this. Try it explicitly:");
            println!("   [[cameras]] model = \"OPENCV_FISHEYE\"   then re-run map and compare.");
        }
    }

    let choices = modelselect::select(&problem, &folds);
    let mut tally: std::collections::BTreeMap<&str, usize> = Default::default();
    for c in &choices {
        *tally.entry(c.recommended).or_insert(0) += 1;
    }

    // Per-camera detail is unreadable at 193 cameras, so show a few and
    // summarise the rest.
    for c in choices.iter().take(3) {
        println!(
            "\ncamera {} ({} observations) -> {}",
            c.camera_id, c.num_observations, c.recommended
        );
        for s in &c.scores {
            println!(
                "   {:<15} {:>2} params   held-out {:>8.4}px   in-sample {:>8.4}px   ({} folds)",
                s.name, s.num_params, s.held_out_px, s.in_sample_px, s.folds
            );
        }
    }
    if choices.len() > 3 {
        println!("\n... and {} more cameras", choices.len() - 3);
    }

    println!("\nRecommended model, over {} camera(s):", choices.len());
    let mut ranked: Vec<(&&str, &usize)> = tally.iter().collect();
    ranked.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (name, n) in &ranked {
        println!("  {name:<15} {n} camera(s)");
    }
    let winner = ranked.first().map(|(n, _)| **n).unwrap_or("SIMPLE_RADIAL");
    println!(
        "elapsed {}",
        progress::human_secs(started.elapsed().as_secs_f64())
    );

    // One entry per camera, because a rig can carry more than one lens and
    // this is exactly the case where a single `images = "*"` rule is wrong.
    let names_of: HashMap<u32, Vec<String>> =
        recon.images.values().fold(HashMap::new(), |mut acc, im| {
            acc.entry(im.camera_id).or_default().push(im.name.clone());
            acc
        });
    let mut blocks = Vec::new();
    let mut ungloballable = Vec::new();
    for c in &choices {
        let mine = names_of.get(&c.camera_id).cloned().unwrap_or_default();
        let others: Vec<String> = names_of
            .iter()
            .filter(|(id, _)| **id != c.camera_id)
            .flat_map(|(_, v)| v.clone())
            .collect();
        match modelselect::glob_for_camera(&mine, &others) {
            Some(glob) => blocks.push((c.camera_id, glob, c.recommended)),
            None => ungloballable.push(c.camera_id),
        }
    }
    if !ungloballable.is_empty() {
        println!(
            "note: {} camera(s) share every path component with another, so no glob can \
             select them individually: {:?}",
            ungloballable.len(),
            &ungloballable[..ungloballable.len().min(8)]
        );
    }

    if args.apply {
        if blocks.is_empty() {
            bail!("no camera could be selected by a glob, so there is nothing to write");
        }
        let path = Project::config_path(&project.root);
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        if existing.contains("[[cameras]]") {
            bail!(
                "{} already declares [[cameras]]; not overwriting. The recommendations are \
                 above - set them there by hand instead.",
                path.display()
            );
        }
        let mut out = format!(
            "{existing}\n# Selected by `sfmtory select-model` on held-out reprojection error,\n\
             # scored over {} fold(s) by {}.\n",
            folds.count, folds.kind
        );
        for (camera_id, glob, model) in &blocks {
            out.push_str(&format!(
                "\n[[cameras]]\nname = \"camera{camera_id}\"\nimages = \"{glob}\"\nmodel = \"{model}\"\n"
            ));
        }
        std::fs::write(&path, out)?;
        println!(
            "Applied {} per-camera entr(y/ies) to {}",
            blocks.len(),
            path.display()
        );
    } else {
        println!("Re-run with --apply to write one [[cameras]] entry per camera.");
        for (camera_id, glob, model) in blocks.iter().take(6) {
            println!("   camera {camera_id}: images = \"{glob}\"  model = \"{model}\"");
        }
    }

    project.record_log(
        "select-model",
        &serde_json::json!({
            "stage": "select-model",
            "status": "ok",
            "fold_kind": folds.kind,
            "num_folds": folds.count,
            "recommended": winner,
            "tally": tally,
            "elapsed_ms": started.elapsed().as_millis(),
        }),
    )?;
    Ok(())
}

fn cmd_dataset_layouts(args: DatasetProjectArgs) -> Result<()> {
    let project = Project::open(&args.project)?;
    let all = project.all_layouts();
    if all.is_empty() {
        println!(
            "{} declares no layouts. Add one with `sfmtory dataset add-layout`.",
            Project::config_path(&project.root).display()
        );
        return Ok(());
    }
    let linked = project.linked_images_dir();
    for l in &all {
        let source = l
            .layout
            .source_dir(&project.root, &project.config.images_dir);
        println!("{}{}", l.name, if l.default { "  (default)" } else { "" });
        println!("  layers  [{}]", l.layout.layers.join(", "));
        println!("  source  {}", source.display());
    }
    println!(
        "\nlink tree: {} ({})",
        linked.display(),
        if linked.is_dir() {
            "built"
        } else {
            "not built"
        }
    );
    Ok(())
}

/// Appends a `[[layouts]]` block, migrating a lone `[layout]` first.
///
/// Editing the file textually rather than re-serialising the parsed config
/// keeps the user's comments and ordering, which a round-trip through `toml`
/// would silently discard.
fn cmd_dataset_add_layout(args: DatasetAddLayoutArgs) -> Result<()> {
    if args.layers.is_empty() {
        bail!("--layers needs at least one role, e.g. --layers capture,camera");
    }
    let project = Project::open(&args.project)?;
    if project.all_layouts().iter().any(|l| l.name == args.name) {
        bail!("this project already has a layout named \"{}\"", args.name);
    }
    let path = Project::config_path(&project.root);
    let mut text = std::fs::read_to_string(&path).unwrap_or_default();

    // A lone `[layout]` and a `[[layouts]]` list cannot both be authoritative,
    // so the first add converts the old form rather than leaving one silently
    // ignored.
    if let Some(existing) = project.config.layout.clone() {
        let start = text.find("[layout]").unwrap_or(text.len());
        let end = text[start..]
            .find("\n[")
            .map(|i| start + i + 1)
            .unwrap_or(text.len());
        text.replace_range(start..end, "");
        text.push_str(&render_layout("default", true, &existing));
        println!("migrated the existing [layout] to [[layouts]] named \"default\"");
    }

    let cfg = project::LayoutConfig {
        source: args.source,
        layers: args.layers,
    };
    // Fail before writing if the roles are not understood.
    layout::validate(&cfg)?;
    text.push_str(&render_layout(&args.name, args.default, &cfg));
    std::fs::write(&path, text)?;
    println!("added layout \"{}\" to {}", args.name, path.display());
    println!(
        "Run `sfmtory dataset link --layout {}` to build its tree.",
        args.name
    );
    Ok(())
}

fn render_layout(name: &str, default: bool, cfg: &project::LayoutConfig) -> String {
    let mut out = format!("\n[[layouts]]\nname = \"{name}\"\n");
    if default {
        out.push_str("default = true\n");
    }
    if let Some(src) = &cfg.source {
        out.push_str(&format!("source = {:?}\n", src.display().to_string()));
    }
    let layers = cfg
        .layers
        .iter()
        .map(|l| format!("{l:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("layers = [{layers}]\n"));
    out
}

fn cmd_dataset_remove_layout(args: DatasetRemoveLayoutArgs) -> Result<()> {
    let project = Project::open(&args.project)?;
    if !project.all_layouts().iter().any(|l| l.name == args.name) {
        bail!(
            "no layout named \"{}\"; this project has: {}",
            args.name,
            project
                .all_layouts()
                .iter()
                .map(|l| l.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let path = Project::config_path(&project.root);
    let text = std::fs::read_to_string(&path)?;
    let Some(updated) = remove_layout_block(&text, &args.name) else {
        bail!(
            "layout \"{}\" is not a [[layouts]] block in {}; edit the file by hand",
            args.name,
            path.display()
        );
    };
    std::fs::write(&path, updated)?;
    println!("removed layout \"{}\" from {}", args.name, path.display());
    Ok(())
}

/// Deletes the `[[layouts]]` block whose `name` matches, textually.
///
/// Returns `None` when no such block is found, so the caller can say so rather
/// than silently writing the file back unchanged.
fn remove_layout_block(text: &str, name: &str) -> Option<String> {
    let needle = format!("name = {name:?}");
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("[[layouts]]") {
        let start = search_from + rel;
        let end = text[start + 1..]
            .find("\n[")
            .map(|i| start + 1 + i + 1)
            .unwrap_or(text.len());
        if text[start..end].lines().any(|l| l.trim() == needle) {
            let mut out = String::with_capacity(text.len());
            out.push_str(&text[..start]);
            out.push_str(&text[end..]);
            // Collapse the blank run the removal leaves behind.
            while out.contains("\n\n\n") {
                out = out.replace("\n\n\n", "\n\n");
            }
            return Some(out);
        }
        search_from = end;
    }
    None
}

fn cmd_dataset_link(args: DatasetLinkArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    if project.all_layouts().is_empty() {
        bail!(
            "{} declares no layout, so there is nothing to normalise.\n\
             Add one when the dataset's directory shape cannot be inferred. `layers` \
             names each level of the path, ending with the file - for a rig dumping one \
             directory per capture and one file per camera:\n\n\
             sfmtory dataset add-layout --name rig --layers capture,camera\n\n\
             Roles: capture, camera, image (a shot within one camera), ignore. A level \
             may also be a pattern such as cam{{camera}}_{{image}}.\n",
            Project::config_path(&project.root).display()
        );
    }
    let named = project.resolve_layout(args.layout.as_deref())?;
    let cfg = named.layout.clone();
    println!("Layout \"{}\"", named.name);
    let source = cfg.source_dir(&project.root, &project.config.images_dir);
    let plan = layout::plan(&source, &cfg)?;

    println!("Source {}", source.display());
    println!(
        "  {} captures x {} cameras -> {} images",
        plan.captures.len(),
        plan.cameras.len(),
        plan.placed.len()
    );
    let max_slot = plan.placed.iter().map(|p| p.image_index).max().unwrap_or(0);
    // Only meaningful when each (capture, camera) holds a single frame; with an
    // image level there are legitimately more images than slots.
    let full = plan.captures.len() * plan.cameras.len();
    if max_slot == 0 && plan.placed.len() != full {
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
    if max_slot > 0 {
        println!("  up to {} shot(s) per camera per capture", max_slot + 1);
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
        "layout": named.name,
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
        let bytes =
            std::fs::read(&d.path).with_context(|| format!("reading {}", d.path.display()))?;
        let img = image::open(&d.path).with_context(|| format!("decoding {}", d.path.display()))?;
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
        let mark = if e.method == result.method {
            "->"
        } else {
            "  "
        };
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
    println!(
        "\nWrote {} and cameras.toml",
        stage_dir.join("intrinsics.json").display()
    );

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
    let reporter = progress::Progress::new("feature", discovered.len());
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
                reporter.tick();
                Ok((w, h, features))
            })
            .collect()
    });
    eprintln!(
        "feature: detection finished in {}",
        progress::human_secs(reporter.elapsed_secs())
    );

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
                db.upsert_camera(&Camera {
                    camera_id: id,
                    model,
                    width: w,
                    height: h,
                })?;
                eprintln!(
                    "camera {id} \"{}\" ({model_name}{}){}",
                    cfg.name,
                    if cfg.params.is_some() {
                        ", known intrinsics"
                    } else {
                        ""
                    },
                    if cfg.refine == Some(false) {
                        ", held fixed"
                    } else {
                        ""
                    }
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
                        model: default_camera_model(&args.camera_model, w, h)
                            .expect("clap restricts this to a known model"),
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
                model: default_camera_model(&args.camera_model, w, h)
                    .expect("clap restricts this to a known model"),
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
                    capture,
                    d.camera_id,
                    d.image_index,
                    marker,
                    corner,
                    image_id,
                    d.name,
                    kp.x,
                    kp.y
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
        let mut out = String::from("feature_id,image_id,image_name,x,y\n");
        out.push_str(&corner_rows.join("\n"));
        out.push('\n');
        std::fs::write(&path, out).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "Wrote {} fiducial corner ids to {}",
            corner_rows.len(),
            path.display()
        );
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
        groups
            .entry((d.camera_id, d.image_index))
            .or_default()
            .push(i);
    }

    let mut out = Vec::with_capacity(groups.len());
    for ((camera_id, image_index), members) in groups {
        let (w, h) = (records[members[0]].1, records[members[0]].2);
        for &m in &members {
            if (records[m].1, records[m].2) != (w, h) {
                bail!(
                    "--merge-multicaps: camera {camera_id} slot {image_index} mixes image sizes \
                     ({}x{} vs {}x{}); merged images must come from the same unmoved camera",
                    w,
                    h,
                    records[m].1,
                    records[m].2
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
                sfm_core::Descriptors::Binary {
                    bytes_per_descriptor,
                    data,
                } => {
                    binary_stride = *bytes_per_descriptor;
                    binary_data.extend_from_slice(data);
                }
            }
        }
        let descriptors = if !marker_data.is_empty() {
            sfm_core::Descriptors::MarkerCorner { data: marker_data }
        } else if !float_data.is_empty() {
            sfm_core::Descriptors::Float32 {
                dim: float_dim,
                data: float_data,
            }
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
            sfm_core::FeatureSet {
                keypoints,
                descriptors,
            },
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
        PairingArg::Exhaustive
            | PairingArg::Sequential
            | PairingArg::VocabTree
            | PairingArg::WithinCapture
    ) {
        bail!(
            "pairing {:?} is not implemented yet (see PLAN.md); available now: exhaustive, \
             sequential, vocab-tree, within-capture",
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
        PairingArg::WithinCapture => {
            let captures = db.list_images_with_capture()?;
            let capture_of: HashMap<u32, i64> =
                captures.iter().map(|(id, _, _, cap)| (*id, *cap)).collect();
            let merged = capture_of.values().any(|c| *c < 0);
            if merged {
                bail!(
                    "--pairing within-capture needs per-capture images, but this project was \
                     built with --merge-multicaps, which pools them. Re-run `feature` without it."
                );
            }
            let pairs: Vec<(usize, usize)> = sfm_match::exhaustive_pairs(n)
                .into_iter()
                .filter(|&(i, j)| capture_of.get(&images[i].0) == capture_of.get(&images[j].0))
                .collect();
            eprintln!(
                "within-capture pairing: {} pairs from {n} images ({:.1}% of exhaustive)",
                pairs.len(),
                100.0 * pairs.len() as f64 / (n * (n - 1) / 2).max(1) as f64
            );
            pairs
        }
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
    // Pair count grows quadratically, so this is the stage most likely to run
    // far longer than expected: 965 images exhaustively is 465k pairs.
    let reporter = progress::Progress::new("match", pairs.len());
    let results: Vec<Option<(usize, usize, sfm_core::TwoViewGeometryRecord)>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let cam_i = &cameras[&images[i].1].model;
            let cam_j = &cameras[&images[j].1].model;
            let out =
                sfm_match::match_and_verify(&features[i], &features[j], cam_i, cam_j, &params)
                    .map(|rec| (i, j, rec));
            reporter.tick();
            out
        })
        .collect();
    eprintln!(
        "match: verification finished in {}",
        progress::human_secs(reporter.elapsed_secs())
    );

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

/// Everything `run_incremental`/`run_global` need, loaded from the project
/// database and config.
///
/// Extracted from `cmd_map` because choosing a camera model honestly requires
/// *rebuilding*: scoring candidate models against structure the incumbent model
/// triangulated cannot see a wrong projection family at all (see
/// `modelselect::radial_residual_trend`). Both callers have to load identically,
/// or the comparison would be measuring the loader rather than the model.
pub struct MapInput {
    pub input: sfm_reconstruction::ReconstructionInput,
    pub num_images: usize,
}

pub fn load_map_input(project: &Project) -> Result<MapInput> {
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
    let known_names: std::collections::HashSet<&str> = project
        .config
        .poses
        .iter()
        .map(|p| p.image.as_str())
        .collect();
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

    // The config is authoritative over the database for camera *models*.
    //
    // `feature` writes each declared camera into the database when it runs, so
    // normally the two already agree and this is a no-op. They disagree exactly
    // when the user edits `[[cameras]]` afterwards - which is the whole workflow
    // for fixing a wrong camera model. Without this, editing `sfm.toml` and
    // re-running `map` changes nothing at all, silently: the reconstruction
    // keeps using the model `feature` recorded, and the only way to apply an
    // edit is to delete the database and redo feature extraction and matching.
    // That cost a real debugging session here, twice.
    let mut cameras = cameras;
    let mut overridden: Vec<u32> = Vec::new();
    for cfg in &project.config.cameras {
        let Some(model_name) = cfg.model.as_deref() else {
            continue;
        };
        let ids: std::collections::BTreeSet<u32> = images
            .iter()
            .filter(|(_, _, name, ..)| project::glob_match(&cfg.images, name))
            .map(|(_, camera_id, ..)| *camera_id)
            .collect();
        for id in ids {
            let Some(existing) = cameras.get(&id) else {
                continue;
            };
            let model = match &cfg.params {
                Some(params) => CameraModel::from_name_and_params(model_name, params)
                    .with_context(|| {
                        format!(
                            "camera \"{}\": model {model_name} does not accept {} parameters",
                            cfg.name,
                            params.len()
                        )
                    })?,
                None => default_camera_model(model_name, existing.width, existing.height)
                    .with_context(|| {
                        format!("camera \"{}\": unknown model {model_name}", cfg.name)
                    })?,
            };
            if model != existing.model {
                overridden.push(id);
                cameras.insert(
                    id,
                    Camera {
                        camera_id: id,
                        model,
                        width: existing.width,
                        height: existing.height,
                    },
                );
            }
        }
    }
    if !overridden.is_empty() {
        overridden.sort_unstable();
        overridden.dedup();
        eprintln!(
            "using the camera model(s) declared in sfm.toml for camera(s) {overridden:?}, \
             which differ from what `feature` recorded"
        );
    }

    let input = sfm_reconstruction::ReconstructionInput {
        images: image_inputs,
        cameras,
        pairs,
        fixed_cameras,
    };
    Ok(MapInput {
        input,
        num_images: images.len(),
    })
}

fn cmd_map(args: MapArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    let stage_dir = project.prepare_stage("map")?;

    let MapInput { input, num_images } = load_map_input(&project)?;

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
        "num_images_input": num_images,
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
        num_images,
        recon.points3d.len(),
        recon.mean_reprojection_error(),
        project.sparse_dir().display(),
        log_path.display()
    );

    // Say when the calibration did not actually happen. A reconstruction that
    // registers every image still reports a focal length for each camera, and
    // nothing in that output distinguishes a refined one from the untouched
    // field-of-view guess it started from - which is how a single-capture rig
    // can look like a successful calibration and be nothing of the kind.
    warn_about_unrefined_intrinsics(&project, &recon);
    Ok(())
}

/// Reports cameras whose intrinsics came out exactly as they went in.
fn warn_about_unrefined_intrinsics(project: &Project, recon: &sfm_core::Reconstruction) {
    let initial = match crate::db::Database::open(&project.database_path())
        .and_then(|db| db.list_cameras())
    {
        Ok(cams) => cams
            .into_iter()
            .map(|c| (c.camera_id, c.model))
            .collect::<std::collections::BTreeMap<_, _>>(),
        Err(_) => return,
    };
    let pinned: std::collections::BTreeSet<u32> = project
        .config
        .cameras
        .iter()
        .filter(|c| c.refine == Some(false))
        .flat_map(|cfg| {
            recon
                .images
                .values()
                .filter(|im| project::glob_match(&cfg.images, &im.name))
                .map(|im| im.camera_id)
        })
        .collect();

    let diag = diagnostics::Diagnostics::compute(recon, &initial, &pinned);
    let stuck: Vec<&diagnostics::CameraDiag> = diag
        .cameras
        .iter()
        .filter(|c| c.verdict.is_warning())
        .collect();
    if stuck.is_empty() {
        return;
    }

    eprintln!();
    eprintln!();
    eprintln!(
        "WARNING: {} of {} camera(s) kept the focal length they started with; not calibrated.",
        stuck.len(),
        diag.cameras.len()
    );
    for c in stuck.iter().take(4) {
        eprintln!("  camera {}: {}", c.camera_id, c.verdict.headline());
        if let Some(e) = c.evidence() {
            eprintln!("      {e}");
        }
    }
    if stuck.len() > 4 {
        eprintln!("  ... and {} more", stuck.len() - 4);
    }
    let single_view = stuck.iter().any(|c| {
        matches!(c.verdict, diagnostics::FocalVerdict::NotEligible { num_images } if num_images < 2)
    });
    if single_view {
        eprintln!("  A camera with one image cannot have its focal recovered from that image:");
        eprintln!("  moving the focal and moving the camera toward or away from the scene");
        eprintln!("  produce almost the same picture, so nothing separates them. Either give");
        eprintln!("  each camera several images of differently-placed targets, or supply known");
        eprintln!("  intrinsics:  [[cameras]] params = [f, cx, cy, 0.0]  with  refine = false");
    }
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
        camera_model: args.camera_model.clone(),
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
