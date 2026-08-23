mod db;
mod project;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use sfm_core::{Camera, CameraModel};

use db::Database;
use project::Project;

#[derive(Parser)]
#[command(
    name = "sfm",
    version,
    about = "Advanced sparse structure-from-motion / camera calibration, no GUI."
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
    /// Stage 1: detect keypoints/descriptors for every image into the project database.
    Extract(ExtractArgs),
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
    #[arg(long)]
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

#[derive(clap::Args)]
struct ExtractArgs {
    #[arg(long)]
    project: PathBuf,
    #[arg(long, value_enum, default_value_t = DetectorArg::Sift)]
    detector: DetectorArg,
    #[arg(long)]
    aruco_dict: Option<String>,
    #[arg(long)]
    max_features: Option<u32>,
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
    #[arg(long)]
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
    #[arg(long)]
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
    #[arg(long)]
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
    #[arg(long)]
    project: PathBuf,
    #[arg(long, value_enum)]
    format: ExportFormatArg,
    /// Defaults to the project's `export/` directory if omitted.
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
    #[arg(long)]
    ours: PathBuf,
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long)]
    gt: Option<PathBuf>,
}

#[derive(clap::Args)]
struct RunArgs {
    #[arg(long)]
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
        Commands::Extract(args) => cmd_extract(args),
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
    println!("Next: sfm extract --project {}", project.root.display());
    Ok(())
}

/// Placeholder for a not-yet-implemented pipeline stage: still opens the
/// project, still records a log entry (so `logs/` reflects every invocation),
/// but reports the real status instead of pretending to have run.
fn not_yet_implemented(
    project: &Project,
    stage: &str,
    planned: &str,
    started: Instant,
) -> Result<()> {
    let payload = serde_json::json!({
        "stage": stage,
        "status": "not_implemented",
        "planned_behavior": planned,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    let log_path = project.record_log(stage, &payload)?;
    eprintln!(
        "`{stage}` is not implemented yet (see PLAN.md). Logged to {}.",
        log_path.display()
    );
    Ok(())
}

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "bmp"];

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn list_images_sorted(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading images directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_image_file(p))
        .collect();
    paths.sort();
    Ok(paths)
}

fn cmd_extract(args: ExtractArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;

    let detector_kind = match args.detector {
        DetectorArg::Sift => sfm_features::DetectorKind::Sift,
        DetectorArg::Orb => sfm_features::DetectorKind::Orb,
        DetectorArg::Aruco => sfm_features::DetectorKind::Aruco,
        other => bail!(
            "detector {other:?} is not implemented yet (see PLAN.md); available now: sift, orb, aruco"
        ),
    };
    if args.gpu {
        eprintln!("note: --gpu has no effect for {:?} (classical CPU detector); GPU is used for the learned detectors once implemented", args.detector);
    }
    let config = sfm_features::DetectorConfig::new(detector_kind)
        .with_max_features(args.max_features.map(|v| v as usize));

    let image_paths = list_images_sorted(&project.config.images_dir)?;
    if image_paths.is_empty() {
        bail!("no images found in {}", project.config.images_dir.display());
    }

    // Decoding + detection is embarrassingly parallel and touches no shared
    // state; only the sqlite writes afterward need to be sequential.
    let detected: Vec<Result<(PathBuf, (u32, u32), sfm_core::FeatureSet)>> = image_paths
        .par_iter()
        .map(|path| {
            let (w, h) = image::image_dimensions(path)
                .with_context(|| format!("reading dimensions of {}", path.display()))?;
            let features = sfm_features::detect_file(path, &config)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("detecting features in {}", path.display()))?;
            Ok((path.clone(), (w, h), features))
        })
        .collect();

    let db = Database::open(&project.database_path())?;
    let mut camera_by_dims: HashMap<(u32, u32), u32> = HashMap::new();
    let mut next_camera_id = db.max_camera_id()? + 1;
    let mut next_image_id = db.max_image_id()? + 1;
    let mut per_image = Vec::new();
    let mut total_features = 0usize;

    for result in detected {
        let (path, (w, h), features) = result?;
        let camera_id = if let Some(id) = camera_by_dims.get(&(w, h)) {
            *id
        } else if let Some(id) = db.find_camera_id_by_dims(w, h)? {
            camera_by_dims.insert((w, h), id);
            id
        } else {
            let id = next_camera_id;
            next_camera_id += 1;
            // Initial focal-length guess (no EXIF/sensor data yet): a wide
            // ~53deg horizontal FOV, the same rule of thumb COLMAP falls back
            // on. Refined for real once `sfm-ba` exists.
            let f = w.max(h) as f64 * 1.2;
            let camera = Camera {
                camera_id: id,
                model: CameraModel::SimpleRadial {
                    f,
                    cx: w as f64 / 2.0,
                    cy: h as f64 / 2.0,
                    k: 0.0,
                },
                width: w,
                height: h,
            };
            db.upsert_camera(&camera)?;
            camera_by_dims.insert((w, h), id);
            id
        };

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());
        let image_id = if let Some(id) = db.find_image_id_by_name(&name)? {
            id
        } else {
            let id = next_image_id;
            next_image_id += 1;
            id
        };
        db.upsert_image(image_id, camera_id, &name, w, h)?;
        db.store_features(
            image_id,
            &format!("{:?}", args.detector).to_lowercase(),
            &features,
        )?;

        total_features += features.len();
        per_image.push(serde_json::json!({
            "image_id": image_id,
            "name": name,
            "num_features": features.len(),
        }));
    }

    let payload = serde_json::json!({
        "stage": "extract",
        "status": "ok",
        "detector": format!("{:?}", args.detector).to_lowercase(),
        "num_images": per_image.len(),
        "total_features": total_features,
        "per_image": per_image,
        "elapsed_ms": started.elapsed().as_millis(),
    });
    let log_path = project.record_log("extract", &payload)?;
    println!(
        "Extracted {total_features} features across {} images ({:?}). Logged to {}.",
        per_image.len(),
        args.detector,
        log_path.display()
    );
    Ok(())
}

fn cmd_match(args: MatchArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;

    if args.matcher != MatcherArg::MnnRatio {
        bail!(
            "matcher {:?} is not implemented yet (see PLAN.md); available now: mnn-ratio",
            args.matcher
        );
    }
    if !matches!(
        args.pairing,
        PairingArg::Exhaustive | PairingArg::Sequential
    ) {
        bail!(
            "pairing {:?} is not implemented yet (see PLAN.md); available now: exhaustive, sequential",
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

    if args.pipeline != PipelineArg::Incremental {
        bail!(
            "pipeline {:?} is not implemented yet (see PLAN.md); available now: incremental",
            args.pipeline
        );
    }

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
    let image_inputs: Vec<sfm_reconstruction::ImageInput> = images
        .iter()
        .map(|(id, camera_id, name, ..)| {
            Ok(sfm_reconstruction::ImageInput {
                image_id: *id,
                camera_id: *camera_id,
                name: name.clone(),
                features: db.load_features(*id)?,
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

    let input = sfm_reconstruction::ReconstructionInput {
        images: image_inputs,
        cameras,
        pairs,
    };
    let recon = sfm_reconstruction::run_incremental(
        &input,
        &sfm_reconstruction::IncrementalParams::default(),
    );

    if recon.images.is_empty() {
        bail!("reconstruction failed to register any images; check `sfm match` results (too few/weak verified pairs?)");
    }

    sfm_io::write_colmap_model(&recon, &project.sparse_dir())
        .map_err(|e| anyhow::anyhow!("writing sparse model: {e}"))?;

    let payload = serde_json::json!({
        "stage": "map",
        "status": "ok",
        "pipeline": "incremental",
        "num_images_input": images.len(),
        "num_images_registered": recon.images.len(),
        "num_points3d": recon.points3d.len(),
        "mean_reprojection_error_px": recon.mean_reprojection_error(),
        "elapsed_ms": started.elapsed().as_millis(),
    });
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

fn cmd_refine(args: RefineArgs) -> Result<()> {
    let started = Instant::now();
    let project = Project::open(&args.project)?;
    not_yet_implemented(
        &project,
        "refine",
        &format!(
            "global bundle adjustment robust_loss={:?} refine_intrinsics={}",
            args.robust_loss, args.refine_intrinsics
        ),
        started,
    )
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

fn cmd_eval(args: EvalArgs) -> Result<()> {
    let recon = sfm_io::read_colmap_model(&args.ours)?;
    println!(
        "Loaded {} images / {} points from {}",
        recon.images.len(),
        recon.points3d.len(),
        args.ours.display()
    );
    println!(
        "mean reprojection error: {:.4}px",
        recon.mean_reprojection_error()
    );
    if let Some(baseline) = &args.baseline {
        let base = sfm_io::read_colmap_model(baseline)?;
        println!(
            "baseline ({}) : {} images / {} points, mean reprojection error {:.4}px",
            baseline.display(),
            base.images.len(),
            base.points3d.len(),
            base.mean_reprojection_error()
        );
        eprintln!(
            "note: pose-accuracy alignment (Umeyama) against the baseline is not implemented yet, see PLAN.md §7"
        );
    }
    if args.gt.is_some() {
        eprintln!("note: ground-truth comparison is not implemented yet, see PLAN.md §7");
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
    println!("`sfm run` chains extract -> match -> map -> export; running each stage now.");
    cmd_extract(ExtractArgs {
        project: args.project.clone(),
        detector: args.detector,
        aruco_dict: None,
        max_features: None,
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
