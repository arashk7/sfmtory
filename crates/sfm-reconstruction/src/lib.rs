//! Two SfM engines sharing one input/output data model (`ReconstructionInput`
//! -> `Reconstruction`): this file's own **incremental** (COLMAP-style)
//! pipeline (`run_incremental`) and the **global** (GLOMAP-style
//! rotation/translation-averaging) pipeline in [`global`] (`run_global`).
//!
//! Incremental is the "robust fallback" pipeline from PLAN.md: seed-pair
//! initialization -> triangulate -> repeatedly register the next-best
//! unregistered image via PnP -> triangulate its new points -> periodic
//! bundle adjustment. Simplifications deliberately kept small and documented
//! rather than hidden:
//! - Seed-pair choice is "most inlier matches", not COLMAP's fuller
//!   well-conditioned-ness score (inlier count + triangulation angle spread).
//! - No iterative re-triangulation/track-merging pass after registration -
//!   points are triangulated once when first formed and only ever refined
//!   (not re-triangulated from scratch) by later bundle adjustment.
//! - Point color is a fixed placeholder, not sampled from the source images.
//! All three are reasonable follow-ups once the end-to-end pipeline is
//! validated against COLMAP (PLAN.md §7), not needed for it to be correct.
//!
//! `run_bundle_adjustment`/`assemble_reconstruction` (below) are shared by
//! both pipelines - see `global`'s module docs for the gauge-fixing
//! precondition `run_global` relies on to reuse them safely.

mod global;
pub use global::{run_global, GlobalParams};

use std::collections::HashMap;

use nalgebra::Vector3;
use sfm_core::{
    Camera, CameraModel, FeatureSet, Image, Point3D, Pose, Reconstruction, TrackElement,
    TwoViewGeometryRecord,
};
use sfm_geometry::{
    essential::to_normalized, pnp_ransac, refine_pose_gauss_newton, reprojection_error_normalized,
    triangulate_normalized, triangulation_angle,
};

/// Minimum images sharing one camera before its intrinsics may be refined at
/// all. Self-calibration needs real diversity in camera motion; below this,
/// a flexible model can lower reprojection error by fitting an unphysical
/// coefficient that "explains away" a wrong focal length rather than
/// correcting it (see `run_bundle_adjustment`). Public so a diagnostic
/// front-end can report *why* a camera was never refined without duplicating
/// - and drifting from - the pipeline's own threshold.
pub const MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS: usize = 5;

/// Observations one camera needs before the *final* pass will refine its
/// intrinsics, when it has too few image rows to pass the image-count gate.
///
/// The image count is the wrong measure for a merged fiducial capture. After
/// `--merge-multicaps` a fixed camera has exactly one image row, and that row
/// holds the target in as many poses as there were captures - which is
/// precisely Zhang's configuration and precisely what self-calibration needs.
/// Counting rows calls it ineligible; counting what the row actually contains
/// does not.
///
/// This applies to the final guarded pass only. The growth-time pass keeps the
/// image-count gate: it has no error comparison and no re-triangulation to
/// recover from a bad step, and ungating it was measured at 5743px mean
/// reprojection error against 0.83px.
const MIN_OBSERVATIONS_FOR_INTRINSICS: usize = 200;

/// How far a refined focal length may move from its starting guess before the
/// solve is treated as diverged rather than calibrated. COLMAP's equivalent
/// (`min/max_focal_length_ratio`) is 0.1x to 10x.
const MIN_FOCAL_RATIO: f64 = 0.25;
const MAX_FOCAL_RATIO: f64 = 4.0;

#[derive(Clone)]
pub struct ImageInput {
    pub image_id: u32,
    pub camera_id: u32,
    pub name: String,
    pub features: FeatureSet,
    /// Known world-to-camera pose, if one was supplied (see the project
    /// config's `[[poses]]`). When at least two images carry one, the
    /// reconstruction starts from those instead of searching for a seed pair.
    pub initial_pose: Option<Pose>,
    /// Whether bundle adjustment may move `initial_pose`. Ignored when
    /// `initial_pose` is `None`.
    pub pose_fixed: bool,
}

#[derive(Clone)]
pub struct PairInput {
    pub i: usize,
    pub j: usize,
    pub geometry: TwoViewGeometryRecord,
}

pub struct ReconstructionInput {
    pub images: Vec<ImageInput>,
    pub cameras: HashMap<u32, Camera>,
    pub pairs: Vec<PairInput>,
    /// Camera ids whose intrinsics must not be refined (see the project
    /// config's `refine = false`).
    pub fixed_cameras: std::collections::HashSet<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct IncrementalParams {
    pub min_triangulation_angle_deg: f64,
    /// Stricter than `min_triangulation_angle_deg`, and used only to *rank*
    /// candidate seed pairs (see `well_conditioned_match_count`). A seed pair
    /// picked by raw inlier count alone tends to be the pair of *most
    /// similar* viewpoints - short baseline relative to scene depth - which
    /// gives every triangulated point a small angle regardless of the
    /// scene's true 3D structure (the classic bas-relief ambiguity: a short
    /// baseline makes even genuinely 3D scenes look nearly flat). A flat-
    /// looking point cloud is exactly what breaks the linear PnP-DLT solver
    /// used to register every subsequent image (see `sfm-geometry::pnp`
    /// module docs), so the seed - and only the seed - needs a real,
    /// wide-baseline check, not just the same lenient bar every other point
    /// triangulation uses.
    pub seed_min_triangulation_angle_deg: f64,
    /// Seed candidates need at least this many raw verified matches, on top
    /// of the angle-based scoring above - otherwise a *sparse* wide-baseline
    /// pair (few matches, but a good fraction of them happen to clear the
    /// angle bar) can outscore a dense, reliable pair whose baseline is
    /// merely narrower, and win seed selection anyway. A seed built from a
    /// handful of points is thin and noise-sensitive regardless of how wide
    /// its baseline nominally is, and produced a visibly worse reconstruction
    /// than a seed picked from a well-matched pair in real testing (see
    /// PLAN.md's real-data-testing entries) - this floor is the fix.
    pub seed_min_matches: usize,
    pub max_reprojection_error_px: f64,
    pub min_pnp_correspondences: usize,
    pub pnp_ransac_threshold_px: f64,
    pub pnp_ransac_max_iterations: usize,
    pub run_ba_every_n_images: usize,
    pub ba_robust_loss: sfm_ba::RobustLoss,
    /// Whether to refine camera intrinsics (focal length, principal point,
    /// distortion) alongside poses/points. On by default: without this, a
    /// camera's intrinsics are whatever the initial width-based guess from
    /// `sfm extract` was, forever - measurably worse calibration than COLMAP
    /// on real test data (see PLAN.md's real-data-testing entry). Exposed as
    /// a flag mainly for isolating intrinsics-refinement bugs during testing.
    pub refine_intrinsics: bool,
}

impl Default for IncrementalParams {
    fn default() -> Self {
        IncrementalParams {
            min_triangulation_angle_deg: 2.0,
            seed_min_triangulation_angle_deg: 6.0,
            seed_min_matches: 100,
            max_reprojection_error_px: 4.0,
            min_pnp_correspondences: 12,
            pnp_ransac_threshold_px: 8.0,
            pnp_ransac_max_iterations: 2000,
            run_ba_every_n_images: 5,
            ba_robust_loss: sfm_ba::RobustLoss::Huber,
            refine_intrinsics: true,
        }
    }
}

#[derive(Clone)]
struct PointWork {
    xyz: Vector3<f64>,
    track: Vec<(usize, u32)>, // (image compact idx, keypoint idx)
}

fn keypoint_px(features: &FeatureSet, idx: u32) -> (f64, f64) {
    let kp = features.keypoints[idx as usize];
    (kp.x as f64, kp.y as f64)
}

/// Look up an image's keypoint index within a stored `(i < j)` pairwise match
/// list, returning `(this_kp, other_kp)` regardless of which side `this_idx`
/// was stored as.
fn oriented_matches(pair: &PairInput, this_idx: usize) -> Option<Vec<(u32, u32)>> {
    if this_idx == pair.i {
        Some(pair.geometry.inlier_matches.clone())
    } else if this_idx == pair.j {
        Some(
            pair.geometry
                .inlier_matches
                .iter()
                .map(|&(a, b)| (b, a))
                .collect(),
        )
    } else {
        None
    }
}

pub fn run_incremental(input: &ReconstructionInput, params: &IncrementalParams) -> Reconstruction {
    let n = input.images.len();
    if n < 2 || input.pairs.is_empty() {
        return Reconstruction::new();
    }

    // Mutable working copy of every camera's intrinsics - starts as the
    // initial guess from `sfm extract`, refined in place by
    // `run_bundle_adjustment` as registration proceeds. Everything from here
    // on (PnP registration, triangulation, the final export) reads *this*
    // map, not `input.cameras`, so refined intrinsics actually take effect
    // instead of being computed and discarded.
    let cameras: HashMap<u32, CameraModel> = input
        .cameras
        .iter()
        .map(|(&id, cam)| (id, cam.model))
        .collect();

    // (i, j) with i < j -> the pair's data, for O(1) neighbor lookups.
    let mut pair_of: HashMap<(usize, usize), &PairInput> = HashMap::new();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for pair in &input.pairs {
        pair_of.insert((pair.i, pair.j), pair);
        neighbors[pair.i].push(pair.j);
        neighbors[pair.j].push(pair.i);
    }

    // Restrict seed candidates to the match graph's largest connected
    // component. A seed picked purely by local pair quality can still be a
    // structural dead end: on `temple_sparse_ring`, an isolated, otherwise
    // disconnected pair with excellent match density and baseline won seed
    // selection outright over every pair in the dataset's real ~13-image
    // component, capping registration at 2/16 no matter what the rest of
    // the pipeline did afterward - no seed placed outside a component can
    // ever grow into it, whatever heuristics run downstream.
    let mut component = vec![usize::MAX; n];
    let mut component_sizes = Vec::new();
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let comp_id = component_sizes.len();
        let mut stack = vec![start];
        component[start] = comp_id;
        let mut size = 0;
        while let Some(node) = stack.pop() {
            size += 1;
            for &nb in &neighbors[node] {
                if component[nb] == usize::MAX {
                    component[nb] = comp_id;
                    stack.push(nb);
                }
            }
        }
        component_sizes.push(size);
    }
    let largest_component = component_sizes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &s)| s)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // The pair with the most inlier matches is very often the pair of *most
    // similar* viewpoints (adjacent photos on a walk, barely moved between
    // shots) - exactly the wrong choice for a seed, since a short baseline
    // relative to scene depth gives every point a tiny triangulation angle
    // and next to nothing survives the angle filter below. Score candidates
    // by how many of their matches would actually triangulate well instead.
    let all_pairs: Vec<&PairInput> = input
        .pairs
        .iter()
        .filter(|p| component[p.i] == largest_component)
        .collect();
    let dense_enough: Vec<&PairInput> = all_pairs
        .iter()
        .copied()
        .filter(|p| p.geometry.inlier_matches.len() >= params.seed_min_matches)
        .collect();
    // Falls back to ranking every pair (ignoring the density floor) only if
    // *no* pair clears it at all - a genuinely small/sparse input set should
    // still get a best-effort seed rather than refusing to reconstruct.
    let candidates: &[&PairInput] = if dense_enough.is_empty() {
        &all_pairs
    } else {
        &dense_enough
    };
    // A seed picked purely by its own match quality can still sit in a
    // sparse branch of an otherwise well-connected graph - COLMAP's own
    // incremental mapper has the identical fallback (`FindNextInitImage`
    // tries multiple candidate pairs) for exactly this reason: no static
    // heuristic reliably predicts how far a *given* seed will actually grow
    // once bridge/bootstrap paths and PnP successes/failures compound down
    // the line, so the only fully reliable test is to try growing from it.
    // Try the best few candidates by the existing quality score and keep
    // whichever genuinely registers the most images.
    let mut ranked: Vec<&PairInput> = candidates.to_vec();
    ranked.sort_by_key(|p| {
        std::cmp::Reverse(well_conditioned_match_count(
            input,
            &cameras,
            p,
            params.seed_min_triangulation_angle_deg,
            params.max_reprojection_error_px,
        ))
    });
    const MAX_SEED_TRIALS: usize = 8;

    // Seed selection, COLMAP-style: grow from the best-ranked candidate and
    // stop as soon as one registers every image in the component - no other
    // seed can beat that, so the remaining candidates are pure waste. Only a
    // seed that leaves images behind causes the next one to be tried, and the
    // best result so far is kept throughout.
    //
    // This replaced growing all `MAX_SEED_TRIALS` candidates and picking the
    // winner. That was ~8x the work in the common case, and parallelizing it
    // across candidates (the previous attempt at fixing the cost) only hid
    // that on an idle 8-core machine while *blocking* the parallelism inside
    // each growth - the Jacobian and Schur passes are themselves rayon-
    // parallel and were being squeezed into one core each. Running a single
    // growth that has the whole pool to itself is both less total work and
    // better wall-clock than eight growths fighting over the pool.
    // Supplied extrinsics make seed selection moot: the initial
    // reconstruction is whatever the caller pinned down, not something to be
    // searched for. Run exactly one growth, anchored on the first known-pose
    // image so bundle adjustment's gauge-fixing image is one whose pose is
    // meaningful in the caller's frame.
    let known_pose_images: Vec<usize> = (0..n)
        .filter(|&i| input.images[i].initial_pose.is_some())
        .collect();
    if known_pose_images.len() >= 2 {
        let anchor = known_pose_images[0];
        eprintln!(
            "using {} supplied camera pose(s) as the initial reconstruction (anchor: {})",
            known_pose_images.len(),
            input.images[anchor].name
        );
        let seed = ranked.first().copied();
        let result = grow_from_seed(
            input,
            params,
            &pair_of,
            &neighbors,
            cameras.clone(),
            anchor,
            seed.map(|p| p.j).unwrap_or(anchor),
            &seed.map(|p| p.geometry.pose).unwrap_or_else(Pose::identity),
            seed.map(|p| p.geometry.inlier_matches.as_slice())
                .unwrap_or(&[]),
        );
        return finish_growth(input, params, result);
    }

    let target_registered = component_sizes[largest_component];
    let mut best: Option<GrowthResult> = None;
    for seed in ranked.iter().take(MAX_SEED_TRIALS) {
        let result = grow_from_seed(
            input,
            params,
            &pair_of,
            &neighbors,
            cameras.clone(),
            seed.i,
            seed.j,
            &seed.geometry.pose,
            &seed.geometry.inlier_matches,
        );
        let count = result.registered.iter().filter(|&&b| b).count();
        let better = best
            .as_ref()
            .map(|b| count > b.registered.iter().filter(|&&x| x).count())
            .unwrap_or(true);
        if better {
            best = Some(result);
        }
        if count >= target_registered {
            break;
        }
    }

    let result = best.unwrap();
    finish_growth(input, params, result)
}

/// The post-growth passes shared by every way of starting a reconstruction:
/// one fixed-intrinsics global bundle to settle the model, then the single
/// authoritative intrinsics-refining pass, then assembly.
fn finish_growth(
    input: &ReconstructionInput,
    params: &IncrementalParams,
    result: GrowthResult,
) -> Reconstruction {
    let GrowthResult {
        seed_i,
        registered,
        mut poses,
        mut points,
        mut cameras,
    } = result;

    // One more fixed-intrinsics pass before the intrinsics-refining one:
    // growth can end on a bootstrap registration (see the bridge-image
    // fallback above), whose pose is inherently less certain than an
    // ordinary RANSAC/GN-verified PnP registration - an immediate BA runs
    // right after each bootstrap step, but the *last* growth step never
    // gets a following correction pass before intrinsics refinement
    // otherwise, letting one bootstrap's residual pose error leak directly
    // into the focal length estimate.
    run_bundle_adjustment(
        input,
        &mut cameras,
        params.ba_robust_loss,
        params.max_reprojection_error_px,
        seed_i,
        &registered,
        &mut poses,
        &mut points,
        IntrinsicsMode::Fixed,
        BaScope::Global,
    );

    // Only this final bundle adjustment is allowed to touch intrinsics.
    // Refining them *during* the loop would mean every periodic call sees a
    // still-changing focal length, so images registered and points
    // triangulated earlier keep getting computed against a stale
    // intermediate calibration - later passes then have to both fix the
    // intrinsics *and* retroactively correct everything built under the old
    // ones, which measurably failed to fully reconverge within a reasonable
    // iteration budget on real test data (see PLAN.md). Doing it once, at
    // the end, from an otherwise-converged state is exactly the scenario
    // `sfm-ba`'s joint-optimization test validates.
    if params.refine_intrinsics {
        refine_intrinsics_iteratively(
            input,
            params,
            seed_i,
            &registered,
            &mut cameras,
            &mut poses,
            &mut points,
        );
    }

    assemble_reconstruction(input, &cameras, &registered, &poses, &points)
}

/// Mean reprojection error over every triangulated observation, in pixels.
fn mean_reprojection(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    poses: &[Option<Pose>],
    points: &[PointWork],
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for p in points {
        for &(img_idx, kp_idx) in &p.track {
            let Some(pose) = poses[img_idx] else { continue };
            let cam = &cameras[&input.images[img_idx].camera_id];
            let pc = pose.transform_point(&p.xyz);
            if pc.z <= 1e-9 {
                continue;
            }
            let (px, py) = cam.project(&pc);
            let (ox, oy) = keypoint_px(&input.images[img_idx].features, kp_idx);
            sum += ((px - ox).powi(2) + (py - oy).powi(2)).sqrt();
            n += 1;
        }
    }
    if n == 0 {
        f64::MAX
    } else {
        sum / n as f64
    }
}

/// Refine intrinsics, re-triangulate, repeat - then keep the result only if it
/// beats leaving the intrinsics alone.
///
/// The single-pass version could not work, and it took a real dataset to see
/// why. Moving a focal length invalidates every 3D point, because each was
/// triangulated under the old one; measured against the *stale* structure the
/// refined intrinsics always score worse, the fixed-vs-free comparison rejects
/// them, and the model never gets the chance to follow. On a 193-camera rig
/// that left 190 focal lengths sitting exactly on their initial guess with
/// every distortion coefficient at zero - not a calibration that was attempted
/// and declined, but one that could not have succeeded.
///
/// COLMAP's incremental mapper does not have this problem because it does not
/// judge a single pass: `ba_global_max_refinements` iterates bundle adjustment
/// against re-triangulated structure until the model stops changing, bounding
/// the result by plausibility (`min/max_focal_length_ratio`,
/// `max_extra_param`) rather than by comparison. This does the same, and then
/// keeps the safety net COLMAP lacks: the whole iterated result is compared
/// against the untouched baseline once at the end, and discarded if it is
/// worse.
fn refine_intrinsics_iteratively(
    input: &ReconstructionInput,
    params: &IncrementalParams,
    seed_i: usize,
    registered: &[bool],
    cameras: &mut HashMap<u32, CameraModel>,
    poses: &mut [Option<Pose>],
    points: &mut [PointWork],
) {
    /// COLMAP's `ba_global_max_refinements` default.
    const MAX_REFINEMENTS: usize = 5;
    /// Relative change in mean reprojection error below which iterating again
    /// is not worth a full global bundle.
    const CONVERGED: f64 = 5e-4;

    let baseline_cameras = cameras.clone();
    let baseline_poses = poses.to_vec();
    let baseline_points = points.to_vec();
    let baseline_error = mean_reprojection(input, cameras, poses, points);

    let mut previous = baseline_error;
    for _ in 0..MAX_REFINEMENTS {
        run_bundle_adjustment(
            input,
            cameras,
            params.ba_robust_loss,
            params.max_reprojection_error_px,
            seed_i,
            registered,
            poses,
            points,
            IntrinsicsMode::FreeBounded,
            BaScope::Global,
        );
        // The step the single-pass version was missing: let the structure
        // follow the intrinsics before judging either.
        for idx in 0..points.len() {
            retriangulate_point(input, cameras, params, poses, points, idx);
        }
        let now = mean_reprojection(input, cameras, poses, points);
        if !now.is_finite() {
            break;
        }
        let converged =
            previous.is_finite() && previous > 0.0 && (previous - now).abs() / previous < CONVERGED;
        previous = now;
        if converged {
            break;
        }
    }

    // The net effect has to be an improvement on doing nothing. Refinement
    // that diverges, or that trades a better focal for a worse model, is not
    // an improvement however plausible its parameters look.
    let refined_error = mean_reprojection(input, cameras, poses, points);
    if !(refined_error < baseline_error) {
        *cameras = baseline_cameras;
        poses.copy_from_slice(&baseline_poses);
        points.clone_from_slice(&baseline_points);
    }
}

struct GrowthResult {
    seed_i: usize,
    registered: Vec<bool>,
    poses: Vec<Option<Pose>>,
    points: Vec<PointWork>,
    cameras: HashMap<u32, CameraModel>,
}

/// Grow a full reconstruction from one candidate seed pair: triangulate the
/// seed, then repeatedly register the next-best unregistered image (via
/// ordinary PnP, falling back to the bridge-image bootstrap when the match
/// graph's local structure can't support PnP - see `run_incremental`'s
/// caller for why both are needed). Takes its own owned `cameras` map so
/// `run_incremental` can try several seeds independently, each from the same
/// unrefined starting intrinsics, without one trial's periodic bundle
/// adjustments contaminating another's.
#[allow(clippy::too_many_arguments)]
fn grow_from_seed(
    input: &ReconstructionInput,
    params: &IncrementalParams,
    pair_of: &HashMap<(usize, usize), &PairInput>,
    neighbors: &[Vec<usize>],
    mut cameras: HashMap<u32, CameraModel>,
    seed_i: usize,
    seed_j: usize,
    seed_pose: &Pose,
    seed_matches: &[(u32, u32)],
) -> GrowthResult {
    let n = input.images.len();
    let mut registered = vec![false; n];
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    let mut points: Vec<PointWork> = Vec::new();
    let mut obs_to_point: HashMap<(usize, u32), usize> = HashMap::new();
    // Correspondence count at the time PnP last failed for this image, if it
    // has failed at all - `None` means never attempted or never failed.
    // Deliberately *not* a permanent exclusion: as more of the scene gets
    // triangulated, an image that failed with too few/too-planar
    // correspondences can pick up enough new ones on a later pass to
    // succeed. Only skip re-attempting while nothing has changed since the
    // last failure (otherwise every failed image gets retried, uselessly,
    // every single iteration).
    let mut failed_at_count: Vec<Option<usize>> = vec![None; n];
    // Bridge pairs (see the fallback below) whose bootstrap attempt already
    // failed validation - permanently excluded, unlike `failed_at_count`,
    // since a bootstrap depends only on the pair's own cached raw matches
    // (which never change), not on how much of the rest of the scene has
    // been triangulated.
    let mut bootstrap_failed: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    // Which images were registered via the bridge bootstrap rather than
    // ordinary RANSAC/GN-verified PnP - used to gate track completion (see
    // `triangulate_and_complete_tracks`'s docs): a bootstrap pose is
    // inherently less certain, and letting its observations extend
    // *other* images' already-trusted points measurably hurt calibration
    // accuracy on `temple_sparse_ring` (a dataset that leans heavily on
    // bootstrap registrations) even at a strict completion threshold,
    // while helping on datasets with few or no bootstrap registrations.
    let mut bootstrap_registered = vec![false; n];

    // Two ways to start. Either the caller supplied known extrinsics for at
    // least two images - in which case those *are* the initial reconstruction,
    // already in a common world frame, and every pair among them can be
    // triangulated directly - or we bootstrap from a single seed pair whose
    // relative pose came from its essential matrix, with the first image
    // defining the world frame.
    let known: Vec<usize> = (0..n)
        .filter(|&i| input.images[i].initial_pose.is_some())
        .collect();
    if known.len() >= 2 {
        for &i in &known {
            poses[i] = input.images[i].initial_pose;
            registered[i] = true;
        }
        // Triangulate across every verified pair among the known-pose images.
        // Unlike the seed-pair case there is no scale ambiguity to resolve:
        // the supplied poses already share a metric world frame, so these
        // points come out at true scale and everything registered later
        // inherits it.
        for pair in &input.pairs {
            if registered[pair.i] && registered[pair.j] {
                triangulate_pair_matches(
                    input,
                    &cameras,
                    params,
                    pair.i,
                    pair.j,
                    &pair.geometry.inlier_matches,
                    &poses,
                    &mut points,
                    &mut obs_to_point,
                );
            }
        }
    } else {
        poses[seed_i] = Some(Pose::identity());
        poses[seed_j] = Some(*seed_pose);
        registered[seed_i] = true;
        registered[seed_j] = true;

        triangulate_pair_matches(
            input,
            &cameras,
            params,
            seed_i,
            seed_j,
            seed_matches,
            &poses,
            &mut points,
            &mut obs_to_point,
        );
    }

    // Model size at the last full/global bundle adjustment - drives the
    // growth-ratio trigger below (COLMAP's `ba_global_*_ratio`).
    let mut images_at_last_global_ba = 2usize;
    let mut points_at_last_global_ba = points.len().max(1);

    loop {
        // Next-best-view: the unregistered image with the most 2D-3D
        // correspondences against already-triangulated points.
        let mut best: Option<(usize, Vec<(usize, u32)>)> = None; // (image_idx, [(point_idx, kp_idx)])
        for u in 0..n {
            if registered[u] {
                continue;
            }
            let mut correspondences: HashMap<u32, usize> = HashMap::new();
            for &r in &neighbors[u] {
                if !registered[r] {
                    continue;
                }
                let (lo, hi) = (u.min(r), u.max(r));
                let Some(pair) = pair_of.get(&(lo, hi)) else {
                    continue;
                };
                let Some(matches) = oriented_matches(pair, u) else {
                    continue;
                };
                for (k_u, k_r) in matches {
                    if let Some(&point_idx) = obs_to_point.get(&(r, k_r)) {
                        correspondences.entry(k_u).or_insert(point_idx);
                    }
                }
            }
            if correspondences.len() < params.min_pnp_correspondences {
                continue;
            }
            // Skip re-attempting a previously-failed image unless it has
            // picked up genuinely new correspondences since then (see
            // `failed_at_count`'s docs) - avoids uselessly re-running PnP on
            // the exact same data every single iteration.
            if let Some(prev_count) = failed_at_count[u] {
                if correspondences.len() <= prev_count {
                    continue;
                }
            }
            let better = best
                .as_ref()
                .map(|(_, c)| correspondences.len() > c.len())
                .unwrap_or(true);
            if better {
                // Sort by keypoint index before handing this to PnP RANSAC:
                // `correspondences` was just built from a `HashMap`, whose
                // iteration order is randomized per-process (Rust's default
                // hasher). PnP RANSAC's minimal samples are drawn by *index*
                // into whatever order it receives, from a fixed internal
                // seed - so an unsorted, run-to-run-random order silently
                // makes the "fixed seed" sample a different actual 6-point
                // subset each run, which for a near-planar point set can be
                // the difference between a well-conditioned and a degenerate
                // minimal sample. That was previously causing real run-to-run
                // variance in registration count (7-9/11 on the same input).
                let mut sorted: Vec<(u32, usize)> = correspondences.into_iter().collect();
                sorted.sort_unstable_by_key(|&(k, _)| k);
                best = Some((u, sorted.into_iter().map(|(k, p)| (p, k)).collect()));
            }
        }

        let Some((u, correspondences)) = best else {
            // Ordinary next-best-view found no unregistered image sharing
            // enough already-triangulated points with a registered
            // neighbor. That's expected once the match graph's *redundant*
            // portion (anything reachable through a triangle - i.e.
            // sharing an already-triangulated point via some third image)
            // is exhausted: a chain-shaped remainder (a "bridge" image
            // connected to the registered set by a single un-triangulated
            // edge, with no triangle at all) can never satisfy that test,
            // no matter how much of the rest of the scene gets
            // triangulated first, since it shares zero already-
            // triangulated points with its only registered neighbor by
            // construction - not a thin correspondence set, an empty one.
            // Try bootstrapping the best such bridge via its cached
            // two-view relative pose to that neighbor instead.
            if let Some(pair) = find_bridge_candidate(&input.pairs, &registered, &bootstrap_failed)
            {
                let (r, u) = if registered[pair.i] {
                    (pair.i, pair.j)
                } else {
                    (pair.j, pair.i)
                };
                if try_bootstrap_bridge_image(
                    input,
                    &cameras,
                    params,
                    pair,
                    u,
                    r,
                    &mut poses,
                    &mut points,
                    &mut obs_to_point,
                    &bootstrap_registered,
                ) {
                    registered[u] = true;
                    failed_at_count[u] = None;
                    bootstrap_registered[u] = true;
                    // Bootstrapped poses carry more uncertainty than an
                    // ordinary RANSAC/GN-verified PnP registration - no
                    // minimum inlier count backs them, and a bootstrap
                    // chained off *another* bootstrap's approximate pose
                    // (no independent correspondences to correct it either)
                    // compounds that error. Pull everything back into
                    // consistency immediately rather than letting error
                    // accumulate across several chained bootstrap steps
                    // before the periodic counter would otherwise trigger.
                    run_bundle_adjustment(
                        input,
                        &mut cameras,
                        params.ba_robust_loss,
                        params.max_reprojection_error_px,
                        seed_i,
                        &registered,
                        &mut poses,
                        &mut points,
                        IntrinsicsMode::Fixed,
                        // Global, not local: a bootstrap pose is the least
                        // trustworthy kind this pipeline produces, and
                        // chained bootstraps compound each other's error -
                        // `temple_sparse_ring` leans heavily on them and has
                        // a documented history of destabilizing when this
                        // correction pass was narrowed (see
                        // `run_bundle_adjustment`'s intrinsics-pass comment).
                        // Bootstraps are rare enough that the full solve is
                        // affordable here.
                        BaScope::Global,
                    );
                    images_at_last_global_ba = registered.iter().filter(|&&b| b).count();
                    points_at_last_global_ba = points.len().max(1);
                } else {
                    bootstrap_failed.insert((pair.i, pair.j));
                }
                continue;
            }
            break;
        };

        let cam = cameras[&input.images[u].camera_id];
        let points3d: Vec<Vector3<f64>> = correspondences
            .iter()
            .map(|&(p, _)| points[p].xyz)
            .collect();
        let points2d_norm: Vec<(f64, f64)> = correspondences
            .iter()
            .map(|&(_, k)| to_normalized(keypoint_px(&input.images[u].features, k), &cam))
            .collect();
        let avg_focal = (cam.focal_lengths().0 + cam.focal_lengths().1) / 2.0;
        let threshold = params.pnp_ransac_threshold_px / avg_focal;

        let Some((pose, inliers)) = pnp_ransac(
            &points3d,
            &points2d_norm,
            threshold,
            params.pnp_ransac_max_iterations,
        ) else {
            failed_at_count[u] = Some(correspondences.len());
            continue;
        };

        poses[u] = Some(pose);
        registered[u] = true;
        for (idx, &(point_idx, k)) in correspondences.iter().enumerate() {
            if inliers[idx] {
                points[point_idx].track.push((u, k));
                obs_to_point.insert((u, k), point_idx);
            }
        }

        for &r in &neighbors[u] {
            if !registered[r] {
                continue;
            }
            let (lo, hi) = (u.min(r), u.max(r));
            let Some(pair) = pair_of.get(&(lo, hi)) else {
                continue;
            };
            let Some(matches) = oriented_matches(pair, u) else {
                continue;
            };
            triangulate_and_complete_tracks(
                input,
                &cameras,
                params,
                u,
                r,
                &matches,
                &poses,
                &mut points,
                &mut obs_to_point,
                &bootstrap_registered,
            );
        }

        // COLMAP's two-tier bundle-adjustment schedule (`IncrementalMapper`):
        // a *local* bundle after every single registration keeps the newly
        // added image and its immediate neighbourhood consistent, which is
        // all that's needed to keep registering further images correctly;
        // full-model *global* bundles are reserved for when the model has
        // actually grown enough for the far side of it to have drifted.
        // Intrinsics stay fixed throughout both (see `run_incremental`'s
        // final call for why).
        run_bundle_adjustment(
            input,
            &mut cameras,
            params.ba_robust_loss,
            params.max_reprojection_error_px,
            seed_i,
            &registered,
            &mut poses,
            &mut points,
            IntrinsicsMode::Fixed,
            BaScope::Local { center: u },
        );

        // Global-bundle trigger, matching COLMAP's `ba_global_images_ratio` /
        // `ba_global_points_ratio` (both 1.1): re-optimize everything once
        // the model has grown ~10% in either images or points since the last
        // global pass. Growth-proportional rather than every-N-images, so
        // large models don't pay for a full solve nearly as often - the
        // schedule that made incremental SfM's cost scale acceptably.
        let num_reg = registered.iter().filter(|&&b| b).count();
        let num_pts = points.len();
        if num_reg as f64 >= images_at_last_global_ba as f64 * 1.1
            || num_pts as f64 >= points_at_last_global_ba as f64 * 1.1
        {
            run_bundle_adjustment(
                input,
                &mut cameras,
                params.ba_robust_loss,
                params.max_reprojection_error_px,
                seed_i,
                &registered,
                &mut poses,
                &mut points,
                // Intrinsics are refined here, progressively, as the model
                // grows - COLMAP's behaviour, and the difference between a
                // focal length that tracks the reconstruction and one that
                // has to be dragged into place in a single jump at the very
                // end. Deferring it entirely to the final pass meant that
                // pass started from the raw `1.2 * max(w, h)` guess with
                // every pose and point already converged to fit that wrong
                // value: measured on `temple_ring`, that one solve took 76
                // iterations and 8.1 of the stage's 15 seconds, over half
                // the total runtime. Refined progressively instead, the same
                // final pass starts from an almost-correct calibration.
                //
                // (An older revision of this pipeline did try refining
                // intrinsics in-loop and reverted it as not reconverging
                // within a sane iteration budget. That was under the
                // every-5-images full-model schedule this file used to have,
                // where "in-loop" meant re-refining constantly against a
                // still-moving model. Here it happens only at the rare
                // growth-triggered global passes, which is a different
                // proposition - and the measurements below back it.)
                if params.refine_intrinsics {
                    IntrinsicsMode::Free
                } else {
                    IntrinsicsMode::Fixed
                },
                BaScope::Global,
            );
            images_at_last_global_ba = num_reg;
            points_at_last_global_ba = num_pts;
        }
    }

    GrowthResult {
        seed_i,
        registered,
        poses,
        points,
        cameras,
    }
}

/// How many of a candidate seed pair's matches would actually survive
/// triangulation (positive depth, sufficient parallax angle, low
/// reprojection error) if `i` were placed at the identity and `j` at the
/// pair's recovered relative pose. Used only to *rank* seed candidates - the
/// real triangulation (which also dedupes against already-claimed
/// observations) happens afterward in `triangulate_pair_matches`.
fn well_conditioned_match_count(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    pair: &PairInput,
    min_angle_deg: f64,
    max_reprojection_error_px: f64,
) -> usize {
    let pose_a = Pose::identity();
    let pose_b = pair.geometry.pose.clone();
    let cam_a = cameras[&input.images[pair.i].camera_id];
    let cam_b = cameras[&input.images[pair.j].camera_id];
    let min_angle = min_angle_deg.to_radians();
    let center_a = pose_a.camera_center();
    let center_b = pose_b.camera_center();
    let avg_focal_a = (cam_a.focal_lengths().0 + cam_a.focal_lengths().1) / 2.0;
    let avg_focal_b = (cam_b.focal_lengths().0 + cam_b.focal_lengths().1) / 2.0;

    pair.geometry
        .inlier_matches
        .iter()
        .filter(|&&(ka, kb)| {
            let obs_a = to_normalized(keypoint_px(&input.images[pair.i].features, ka), &cam_a);
            let obs_b = to_normalized(keypoint_px(&input.images[pair.j].features, kb), &cam_b);
            let Some(xyz) = triangulate_normalized(&[(pose_a.clone(), obs_a), (pose_b.clone(), obs_b)]) else {
                return false;
            };
            // See `triangulate_pair_matches` for why this is negated: a
            // degenerate pair yields a NaN angle, which must not score as
            // well-conditioned.
            if !(triangulation_angle(&center_a, &center_b, &xyz) >= min_angle) {
                return false;
            }
            let err_a = reprojection_error_normalized(&pose_a, &xyz, obs_a).map(|e| e * avg_focal_a);
            let err_b = reprojection_error_normalized(&pose_b, &xyz, obs_b).map(|e| e * avg_focal_b);
            matches!((err_a, err_b), (Some(a), Some(b)) if a <= max_reprojection_error_px && b <= max_reprojection_error_px)
        })
        .count()
}

/// Best not-yet-excluded pair straddling the registered/unregistered
/// boundary (exactly one endpoint registered), ranked by raw inlier match
/// count - the most trustworthy signal available for a pair we can't yet
/// judge by triangulated-point overlap.
fn find_bridge_candidate<'a>(
    pairs: &'a [PairInput],
    registered: &[bool],
    excluded: &std::collections::HashSet<(usize, usize)>,
) -> Option<&'a PairInput> {
    pairs
        .iter()
        .filter(|p| registered[p.i] != registered[p.j])
        .filter(|p| !excluded.contains(&(p.i, p.j)))
        .max_by_key(|p| p.geometry.inlier_matches.len())
}

/// Median depth (in `r`'s own camera frame) of `r`'s already-triangulated
/// points - a same-order-of-magnitude scene-scale guess for bootstrapping a
/// neighbor's pose, since the two-view relative pose's translation carries
/// only an arbitrary unit-length scale rather than the reconstruction's
/// established one. Used only as a fallback when fewer than two cameras are
/// registered yet (see `median_camera_baseline`, which is the better-
/// matched quantity for a camera-to-camera baseline and is preferred
/// whenever there's enough registered geometry to compute it).
fn median_point_depth(points: &[PointWork], r: usize, r_pose: &Pose) -> Option<f64> {
    let mut depths: Vec<f64> = points
        .iter()
        .filter(|p| p.track.iter().any(|&(img, _)| img == r))
        .map(|p| r_pose.transform_point(&p.xyz).z)
        .filter(|&z| z > 1e-6)
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(depths[depths.len() / 2])
}

/// Median pairwise distance between already-registered camera centers - the
/// right order-of-magnitude scale for bootstrapping *another* camera's
/// position, unlike `median_point_depth` (a camera-to-scene quantity that
/// can differ from camera-to-camera baseline by an order of magnitude for,
/// e.g., a turntable/orbital capture of a small object - exactly
/// `temple_sparse_ring`'s geometry, where using scene depth as the baseline
/// guess badly overestimated it and made otherwise-good bootstrap poses
/// fail triangulation-angle validation).
fn median_camera_baseline(poses: &[Option<Pose>]) -> Option<f64> {
    let centers: Vec<Vector3<f64>> = poses
        .iter()
        .filter_map(|p| p.as_ref().map(Pose::camera_center))
        .collect();
    if centers.len() < 2 {
        return None;
    }
    let mut dists = Vec::new();
    for i in 0..centers.len() {
        for c in &centers[i + 1..] {
            dists.push((centers[i] - c).norm());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(dists[dists.len() / 2])
}

/// Absolute pose candidate for `u`, composed from `r`'s already-known
/// absolute pose and the cached two-view relative pose between `u` and `r`
/// (computed and RANSAC-verified back at match time). Rotation is exact;
/// translation direction is exact but its magnitude is rescaled to
/// `scale_guess` since the two-view estimate's own translation length is
/// an arbitrary unit baseline, not the reconstruction's established scale.
fn compose_pose_via_neighbor(pair: &PairInput, r: usize, r_pose: &Pose, scale_guess: f64) -> Pose {
    let rel = &pair.geometry.pose;
    let norm = rel.translation.norm();
    let rel_t = if norm > 1e-9 {
        rel.translation / norm * scale_guess
    } else {
        rel.translation
    };

    if r == pair.i {
        // rel: X_u = rel.R * X_r + rel.t, with X_r meaning "r's camera
        // frame treated as world" - substitute r's real absolute pose.
        let rotation = rel.rotation * r_pose.rotation;
        let translation = rel.rotation * r_pose.translation + rel_t;
        Pose::from_rotation_translation(rotation, translation)
    } else {
        debug_assert_eq!(r, pair.j);
        // rel: X_r = rel.R * X_u + rel.t  =>  X_u = rel.R^T * X_r - rel.R^T * rel.t
        let rel_inv = rel.rotation.inverse();
        let rotation = rel_inv * r_pose.rotation;
        let translation = rel_inv * r_pose.translation - rel_inv * rel_t;
        Pose::from_rotation_translation(rotation, translation)
    }
}

/// Registers bridge image `u` against its sole registered neighbor `r` via
/// `pair`'s cached two-view geometry rather than PnP-from-3D-points (see the
/// call site in `run_incremental` for why ordinary PnP structurally can't
/// handle this case). Validated by how much of the pair's own raw matches
/// this pose then lets triangulate cleanly (positive depth, real parallax,
/// low reprojection error) - this confirms the bootstrap's rotation and
/// translation *direction* are sound (a sign/orientation blunder would
/// triangulate almost nothing), though it can't confirm absolute scale,
/// since two-view triangulation is self-consistent for any positive scale
/// of the baseline. Rolls back cleanly and returns `false` if validation
/// fails, so a bad candidate never pollutes the map.
#[allow(clippy::too_many_arguments)]
fn try_bootstrap_bridge_image(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    params: &IncrementalParams,
    pair: &PairInput,
    u: usize,
    r: usize,
    poses: &mut [Option<Pose>],
    points: &mut Vec<PointWork>,
    obs_to_point: &mut HashMap<(usize, u32), usize>,
    bootstrap_registered: &[bool],
) -> bool {
    let r_pose = poses[r].expect("neighbor is registered");
    let base_scale = median_camera_baseline(poses)
        .or_else(|| median_point_depth(points, r, &r_pose))
        .unwrap_or(1.0);

    // `u` isn't marked bootstrap-registered in the caller's array yet (that
    // only happens once this function returns `true`), but it unavoidably
    // will be - treat it as such for `triangulate_and_complete_tracks`'s
    // gating purposes regardless of the outcome here.
    let mut bootstrap_registered_with_u = bootstrap_registered.to_vec();
    bootstrap_registered_with_u[u] = true;

    let Some(matches) = oriented_matches(pair, u) else {
        return false;
    };
    let cam_u = cameras[&input.images[u].camera_id];

    // Any of `u`'s matches with `r` that happen to already have a
    // triangulated 3D point (visible via some third, already-registered
    // image) let us nonlinearly correct the bootstrap's scale and any
    // small rotation/translation error before trusting it - same solver
    // ordinary PnP uses to polish its own linear estimate, just invoked
    // directly since a bridge image structurally has too few of these to
    // run RANSAC/DLT at all (that's exactly why it needed this fallback).
    let existing: Vec<(Vector3<f64>, (f64, f64))> = matches
        .iter()
        .filter_map(|&(k_u, k_r)| {
            let point_idx = *obs_to_point.get(&(r, k_r))?;
            let obs = to_normalized(keypoint_px(&input.images[u].features, k_u), &cam_u);
            Some((points[point_idx].xyz, obs))
        })
        .collect();

    let min_gain = ((matches.len() as f64 * 0.1).ceil() as usize)
        .max(3)
        .min(matches.len());

    // `base_scale` is only an order-of-magnitude guess (the relative pose's
    // own translation carries no scale information at all), and the
    // triangulation-angle validation below is scale-*sensitive* - a
    // baseline placed too near or far relative to the true one collapses
    // or distorts every triangulated angle even when rotation and
    // direction are exactly right. Sweep a small multiplier grid around
    // the guess rather than betting everything on it being correct;
    // nonlinear refinement against `existing` (when available) then cleans
    // up whatever residual error the winning multiplier still has.
    for multiplier in [1.0, 0.3, 3.0, 0.1, 10.0, 0.03, 30.0, 0.01, 100.0] {
        let scale_guess = base_scale * multiplier;
        let candidate = compose_pose_via_neighbor(pair, r, &r_pose, scale_guess);
        let pose = if existing.len() >= 3 {
            let (p3, p2): (Vec<_>, Vec<_>) = existing.iter().cloned().unzip();
            refine_pose_gauss_newton(&candidate, &p3, &p2, 15)
        } else {
            candidate
        };

        poses[u] = Some(pose);
        let before = points.len();
        let (gained, completions) = triangulate_and_complete_tracks(
            input,
            cameras,
            params,
            u,
            r,
            &matches,
            poses,
            points,
            obs_to_point,
            &bootstrap_registered_with_u,
        );

        if gained >= min_gain {
            return true;
        }
        for (img, kp, point_idx) in completions {
            points[point_idx]
                .track
                .retain(|&(i, k)| !(i == img && k == kp));
            obs_to_point.remove(&(img, kp));
            // The completion's own position update (if any) was computed
            // including the observation just removed above - recompute
            // from what's actually left so a rejected bootstrap trial can
            // never leave a point's `xyz` reflecting an undone observation.
            retriangulate_point(input, cameras, params, poses, points, point_idx);
        }
        for new_point in &points[before..] {
            for &(img, kp) in &new_point.track {
                obs_to_point.remove(&(img, kp));
            }
        }
        points.truncate(before);
        poses[u] = None;
    }
    false
}

/// Shared by `triangulate_and_complete_tracks` (gating a new observation
/// added to an existing point) and `retriangulate_point` (gating a wholesale
/// position update from a point's full track) - deliberately much stricter
/// than `IncrementalParams::max_reprojection_error_px` (the bar for creating
/// a brand new point from a fresh 2-view match): both mutate an *already-
/// trusted* point in place rather than living or dying on their own, so a
/// marginal case doesn't fail cleanly the way a marginal new point does - it
/// degrades a point every other observation already relies on. Tuned
/// empirically against all three real test datasets, together with
/// `triangulate_and_complete_tracks`'s `bootstrap_registered` gate (also
/// required - even at this threshold, allowing completions *from*
/// bootstrap-registered cameras measurably hurt `temple_sparse_ring`'s
/// focal-length accuracy, a dataset that leans heavily on bootstrap
/// registrations): the 4px fresh-triangulation bar regressed all three
/// datasets outright (worst case, `temple_sparse_ring`: reprojection error
/// 0.38px -> 0.72px, intrinsics rejected entirely); 0.5px improves
/// `sceaux_castle` (3.1% -> 2.78% focal error) and `temple_ring` (1.2% ->
/// 0.99%) while leaving `temple_sparse_ring` within noise of its prior
/// result (3.7% -> 3.86%).
const COMPLETION_MAX_REPROJECTION_ERROR_PX: f64 = 0.5;

/// The completion bar to apply to one image's features.
///
/// The strict `COMPLETION_MAX_REPROJECTION_ERROR_PX` exists to stop a
/// *mis-corresponded* observation from being welded onto an existing point,
/// where the reprojection error is the only available evidence that the match
/// was wrong. Fiducial corners are matched by exact identity, so that failure
/// mode does not exist for them: `(capture, marker, corner)` either matches or
/// it does not, and a wrong point can never be proposed. Holding them to a
/// bar meant for ambiguous correspondences only rejects *correct* completions,
/// because a marker corner comes from a quad fit and is inherently coarser
/// than a subpixel-refined SIFT keypoint - measured on real fiducial photos,
/// this left every single track at length two and, with no multi-view
/// redundancy anywhere in the model, no way to observe the shared focal
/// length. They get the pipeline's ordinary observation bar instead, which
/// still rejects genuinely bad geometry.
/// Whether a correspondence in these features is proven by identity rather
/// than inferred from appearance.
///
/// Fiducial corners match on an exact `(capture, marker, corner)` key, so a
/// *wrong* correspondence is not possible - the identity either matches or it
/// does not. The reprojection gate on track completion exists to catch
/// mis-correspondence, and that failure mode does not exist here.
///
/// Gating them anyway is actively harmful, and measurably so. On a 4-camera
/// rig, 472 corners were detected by all four cameras and the reconstruction
/// produced **no track longer than three**: the initial focal guess was far
/// enough off that an already-triangulated point reprojected into a
/// newly-registered image well outside the bar, so every fourth observation
/// was refused. That is exactly backwards - accepting the observation and
/// re-triangulating is what *corrects* a point placed with a poor focal, and
/// refusing it is what freezes the error in place and leaves nothing long
/// enough to recover the focal from.
fn correspondence_is_proven(features: &FeatureSet) -> bool {
    matches!(
        features.descriptors,
        sfm_core::Descriptors::MarkerCorner { .. }
    )
}

fn completion_threshold_px(features: &FeatureSet, max_reprojection_error_px: f64) -> f64 {
    if matches!(
        features.descriptors,
        sfm_core::Descriptors::MarkerCorner { .. }
    ) {
        max_reprojection_error_px
    } else {
        COMPLETION_MAX_REPROJECTION_ERROR_PX
    }
}

/// Recomputes `points[point_idx]`'s 3D position from its *entire* current
/// track (every current observation, using each observation's current
/// pose) - not just the two views that originally triangulated it. Closes
/// the literal gap `triangulate_and_complete_tracks` otherwise leaves open:
/// a completion adds a *new* observation to a point's track, but until this
/// function is called against it, the point's actual `xyz` stays frozen at
/// whatever the original two-view estimate produced - the new (and every
/// other) observation only ever reaches bundle adjustment as a reprojection
/// target, never as evidence for a better triangulated position.
///
/// Replaces `xyz` only if the fresh N-view triangulation succeeds, clears
/// `min_triangulation_angle_deg`, and every observation reprojects within
/// `COMPLETION_MAX_REPROJECTION_ERROR_PX` (the same strict bar completions
/// themselves use, not the looser `max_reprojection_error_px` fresh two-view
/// triangulation uses) - this mutates an *already-established* point's
/// position wholesale, a stronger action than a completion's single added
/// observation, so it gets at least as strict a bar. A retriangulation that
/// fails these is rejected outright, keeping the point's previous (still
/// valid) position rather than replacing a known-good point with a worse one.
fn retriangulate_point(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    params: &IncrementalParams,
    poses: &[Option<Pose>],
    points: &mut [PointWork],
    point_idx: usize,
) {
    if points[point_idx].track.len() < 2 {
        return;
    }

    let views: Vec<(Pose, (f64, f64), usize)> = points[point_idx]
        .track
        .iter()
        .filter_map(|&(img_idx, kp_idx)| {
            let pose = poses[img_idx]?;
            let cam = &cameras[&input.images[img_idx].camera_id];
            let obs = to_normalized(keypoint_px(&input.images[img_idx].features, kp_idx), cam);
            Some((pose, obs, img_idx))
        })
        .collect();
    if views.len() < 2 {
        return;
    }

    let dlt_input: Vec<(Pose, (f64, f64))> = views.iter().map(|&(p, o, _)| (p, o)).collect();
    let Some(xyz) = triangulate_normalized(&dlt_input) else {
        return;
    };

    let min_angle = params.min_triangulation_angle_deg.to_radians();
    let centers: Vec<Vector3<f64>> = views.iter().map(|&(p, _, _)| p.camera_center()).collect();
    let mut max_angle = 0.0_f64;
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            max_angle = max_angle.max(triangulation_angle(&centers[i], &centers[j], &xyz));
        }
    }
    if max_angle < min_angle {
        return;
    }

    for &(pose, obs, img_idx) in &views {
        let cam = &cameras[&input.images[img_idx].camera_id];
        let avg_focal = (cam.focal_lengths().0 + cam.focal_lengths().1) / 2.0;
        let Some(err) = reprojection_error_normalized(&pose, &xyz, obs) else {
            return;
        };
        if err * avg_focal
            > completion_threshold_px(
                &input.images[img_idx].features,
                params.max_reprojection_error_px,
            )
        {
            return;
        }
    }

    points[point_idx].xyz = xyz;
}

/// For each match in `matches`, either extends an already-triangulated
/// point's track (if exactly one side already maps to a point, and the new
/// observation reprojects that point's *already-established* 3D position
/// within `max_reprojection_error_px` - verified before trusting it, since
/// the claiming side's own pose may not be independently confirmed, e.g.
/// mid bridge-bootstrap) or hands the match to `triangulate_pair_matches`
/// as a candidate for a brand new point (if neither side is claimed yet).
/// Matches where the "new" side already belongs to a *different* point are
/// left alone (rare - a genuinely ambiguous/duplicate match).
///
/// This closes a real gap: ordinary per-pair triangulation previously only
/// ever created points from *fresh* (both-sides-unclaimed) matches, silently
/// discarding any match where the far side already had a point instead of
/// using it to extend that point's track with one more observation - which
/// meant many real, additional viewing angles of already-triangulated
/// points never reached bundle adjustment at all. More viewing angles per
/// point is exactly what a self-calibration solve needs to be well-
/// conditioned (see decisions.md's "Known open gaps").
///
/// Returns `(gained, completions)`: `gained` is the total number of new
/// observations successfully added (completions plus new points' own
/// tracks) - a stronger validation signal than raw triangulation success
/// for e.g. the bridge bootstrap, since a completion is checked against an
/// *independently already-scaled* 3D position, unlike fresh two-view
/// triangulation, which is self-consistent for any positive baseline scale
/// and so can't by itself confirm a bootstrap pose's scale is right.
/// `completions` lists exactly what was added, `(image, keypoint,
/// point_idx)`, so a caller that needs to roll back a rejected attempt
/// (again, the bootstrap path) can undo precisely these track mutations -
/// unlike new points, which a simple `Vec::truncate` handles.
#[allow(clippy::too_many_arguments)]
fn triangulate_and_complete_tracks(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    params: &IncrementalParams,
    a: usize,
    b: usize,
    matches: &[(u32, u32)],
    poses: &[Option<Pose>],
    points: &mut Vec<PointWork>,
    obs_to_point: &mut HashMap<(usize, u32), usize>,
    bootstrap_registered: &[bool],
) -> (usize, Vec<(usize, u32, usize)>) {
    let (Some(pose_a), Some(pose_b)) = (poses[a], poses[b]) else {
        return (0, Vec::new());
    };
    let cam_a = cameras[&input.images[a].camera_id];
    let cam_b = cameras[&input.images[b].camera_id];
    let avg_focal_a = (cam_a.focal_lengths().0 + cam_a.focal_lengths().1) / 2.0;
    let avg_focal_b = (cam_b.focal_lengths().0 + cam_b.focal_lengths().1) / 2.0;

    let mut fresh: Vec<(u32, u32)> = Vec::new();
    let mut completions: Vec<(usize, u32, usize)> = Vec::new();
    for &(ka, kb) in matches {
        let claim_a = obs_to_point.get(&(a, ka)).copied();
        let claim_b = obs_to_point.get(&(b, kb)).copied();
        match (claim_a, claim_b) {
            (None, None) => fresh.push((ka, kb)),
            (Some(point_idx), None) if !bootstrap_registered[b] => {
                let obs_b = to_normalized(keypoint_px(&input.images[b].features, kb), &cam_b);
                let proven = correspondence_is_proven(&input.images[b].features);
                if let Some(err) =
                    reprojection_error_normalized(&pose_b, &points[point_idx].xyz, obs_b)
                {
                    if proven
                        || err * avg_focal_b
                            <= completion_threshold_px(
                                &input.images[b].features,
                                params.max_reprojection_error_px,
                            )
                    {
                        points[point_idx].track.push((b, kb));
                        obs_to_point.insert((b, kb), point_idx);
                        completions.push((b, kb, point_idx));
                        retriangulate_point(input, cameras, params, poses, points, point_idx);
                    }
                }
            }
            (None, Some(point_idx)) if !bootstrap_registered[a] => {
                let obs_a = to_normalized(keypoint_px(&input.images[a].features, ka), &cam_a);
                let proven = correspondence_is_proven(&input.images[a].features);
                if let Some(err) =
                    reprojection_error_normalized(&pose_a, &points[point_idx].xyz, obs_a)
                {
                    if proven
                        || err * avg_focal_a
                            <= completion_threshold_px(
                                &input.images[a].features,
                                params.max_reprojection_error_px,
                            )
                    {
                        points[point_idx].track.push((a, ka));
                        obs_to_point.insert((a, ka), point_idx);
                        completions.push((a, ka, point_idx));
                        retriangulate_point(input, cameras, params, poses, points, point_idx);
                    }
                }
            }
            // Catches both `(Some(_), Some(_))` (both sides already
            // claimed, possibly by different points) and a completion
            // whose guard above rejected it for being bootstrap-sourced -
            // in both cases, leave the match alone rather than triangulate
            // or complete anything from it.
            _ => {}
        }
    }

    let before = points.len();
    triangulate_pair_matches(
        input,
        cameras,
        params,
        a,
        b,
        &fresh,
        poses,
        points,
        obs_to_point,
    );
    let new_points = points.len() - before;

    (completions.len() + new_points, completions)
}

#[allow(clippy::too_many_arguments)]
fn triangulate_pair_matches(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    params: &IncrementalParams,
    a: usize,
    b: usize,
    matches: &[(u32, u32)],
    poses: &[Option<Pose>],
    points: &mut Vec<PointWork>,
    obs_to_point: &mut HashMap<(usize, u32), usize>,
) {
    let (Some(pose_a), Some(pose_b)) = (poses[a].clone(), poses[b].clone()) else {
        return;
    };
    let cam_a = cameras[&input.images[a].camera_id];
    let cam_b = cameras[&input.images[b].camera_id];
    let min_angle = params.min_triangulation_angle_deg.to_radians();
    let center_a = pose_a.camera_center();
    let center_b = pose_b.camera_center();

    for &(ka, kb) in matches {
        if obs_to_point.contains_key(&(a, ka)) || obs_to_point.contains_key(&(b, kb)) {
            continue;
        }
        let obs_a = to_normalized(keypoint_px(&input.images[a].features, ka), &cam_a);
        let obs_b = to_normalized(keypoint_px(&input.images[b].features, kb), &cam_b);
        let Some(xyz) = triangulate_normalized(&[(pose_a.clone(), obs_a), (pose_b.clone(), obs_b)])
        else {
            continue;
        };

        let angle = triangulation_angle(&center_a, &center_b, &xyz);
        // `!(angle >= min_angle)` rather than `angle < min_angle`: the angle is
        // NaN when the two camera centres coincide, which is exactly what a
        // degenerate two-view estimate produces (a recovered relative pose
        // with no translation). Written the other way round, NaN compares
        // false and slips *through* the gate.
        if !(angle >= min_angle) {
            continue;
        }
        let avg_focal_a = (cam_a.focal_lengths().0 + cam_a.focal_lengths().1) / 2.0;
        let avg_focal_b = (cam_b.focal_lengths().0 + cam_b.focal_lengths().1) / 2.0;
        let err_a = reprojection_error_normalized(&pose_a, &xyz, obs_a).map(|e| e * avg_focal_a);
        let err_b = reprojection_error_normalized(&pose_b, &xyz, obs_b).map(|e| e * avg_focal_b);
        let (Some(err_a), Some(err_b)) = (err_a, err_b) else {
            continue;
        };
        if err_a > params.max_reprojection_error_px || err_b > params.max_reprojection_error_px {
            continue;
        }

        let point_idx = points.len();
        points.push(PointWork {
            xyz,
            track: vec![(a, ka), (b, kb)],
        });
        obs_to_point.insert((a, ka), point_idx);
        obs_to_point.insert((b, kb), point_idx);
    }
}

/// Which slice of the reconstruction a `run_bundle_adjustment` call optimizes.
///
/// Replicates COLMAP's `IncrementalMapper::AdjustLocalBundle` /
/// `AdjustGlobalBundle` split, which is the single largest reason its mapping
/// stage is fast: re-optimizing *every* registered image and *every* point
/// after each new registration is quadratic work over the course of a
/// reconstruction, and almost all of it is redundant - registering image 40
/// barely moves image 3's pose. Measured here before the split existed:
/// 130 of 171 CPU-seconds on `temple_ring` went into building Schur
/// complements for repeated full-model solves during growth.
#[derive(Clone, Copy)]
enum BaScope {
    /// Every registered image and every usable point are free variables
    /// (except the gauge-fixing seed). Used for the periodic full
    /// re-optimizations and the final refinement passes.
    Global,
    /// COLMAP's local bundle: only the just-registered image and its most
    /// co-visible neighbours are free, only points that image actually
    /// observes are free, and every *other* image observing those points is
    /// included but held fixed so it still constrains them. Keeps the solve
    /// proportional to the newly-added information rather than to the size
    /// of the whole model.
    Local { center: usize },
}

/// How camera intrinsics are treated by a `run_bundle_adjustment` call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IntrinsicsMode {
    /// Intrinsics held at their current values.
    Fixed,
    /// Intrinsics free, optimized jointly with poses/points in a single plain
    /// solve - no outlier filtering, no fixed-vs-free safety comparison.
    /// Used by the growth-triggered global bundles so the focal length tracks
    /// the model as it grows instead of being corrected in one huge jump at
    /// the end (see `run_incremental`).
    Free,
    /// Intrinsics free, plus outlier filtering and the fixed-vs-free
    /// reprojection-error comparison that can reject the refined result
    /// outright. The final, authoritative calibration pass.
    FreeGuarded,
    /// Intrinsics free with outlier filtering and the plausibility bounds, but
    /// *without* the fixed-vs-free comparison.
    ///
    /// Exists so a caller can iterate refinement and re-triangulation and
    /// judge the result once at the end, which is what COLMAP does
    /// (`ba_global_max_refinements`). The comparison inside `FreeGuarded`
    /// cannot be used that way: it re-judges after every pass, against
    /// structure triangulated under the *old* intrinsics, so a focal that
    /// moves always looks worse and is always rejected - the model never gets
    /// the chance to follow it. See `refine_intrinsics_iteratively`.
    FreeBounded,
}

/// How many co-visible neighbours join the just-registered image as free
/// variables in a local bundle. COLMAP's own `ba_local_num_images` default.
const LOCAL_BA_NUM_IMAGES: usize = 6;

#[allow(clippy::too_many_arguments)]
fn run_bundle_adjustment(
    input: &ReconstructionInput,
    cameras: &mut HashMap<u32, CameraModel>,
    ba_robust_loss: sfm_ba::RobustLoss,
    max_reprojection_error_px: f64,
    seed_i: usize,
    registered: &[bool],
    poses: &mut [Option<Pose>],
    points: &mut [PointWork],
    intrinsics: IntrinsicsMode,
    scope: BaScope,
) {
    let allow_intrinsics = intrinsics != IntrinsicsMode::Fixed;
    // Which points are free variables, and which images are free to move.
    // `Global` frees everything; `Local` frees only what the newly-registered
    // image touches (see `BaScope`).
    let (point_ids, free_images): (Vec<usize>, Option<std::collections::HashSet<usize>>) =
        match scope {
            BaScope::Global => (
                (0..points.len())
                    .filter(|&p| points[p].track.len() >= 2)
                    .collect(),
                None,
            ),
            BaScope::Local { center } => {
                let point_ids: Vec<usize> = (0..points.len())
                    .filter(|&p| {
                        points[p].track.len() >= 2
                            && points[p].track.iter().any(|&(img, _)| img == center)
                    })
                    .collect();
                // Co-visibility = number of `center`'s own points each other
                // registered image also observes. Ranked descending, so the
                // window is the neighbourhood that actually shares structure
                // with the new image rather than an arbitrary index window.
                let mut covis: HashMap<usize, usize> = HashMap::new();
                for &p in &point_ids {
                    for &(img, _) in &points[p].track {
                        if img != center && registered[img] {
                            *covis.entry(img).or_insert(0) += 1;
                        }
                    }
                }
                let mut ranked: Vec<(usize, usize)> = covis.into_iter().collect();
                // Sort by shared-point count, breaking ties by image index so
                // the window is deterministic run to run (`HashMap` iteration
                // order is randomized per process - the same class of bug
                // already fixed once in this pipeline's PnP sampling).
                ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let mut free: std::collections::HashSet<usize> = ranked
                    .into_iter()
                    .take(LOCAL_BA_NUM_IMAGES)
                    .map(|(img, _)| img)
                    .collect();
                free.insert(center);
                (point_ids, Some(free))
            }
        };

    // Images in the problem: every registered image observing a free point.
    // For `Global` that's just "every registered image"; for `Local` the ones
    // outside `free_images` come along as *fixed* poses - they still pull on
    // the points they observe, they just don't move.
    let image_ids: Vec<usize> = match &free_images {
        None => (0..registered.len()).filter(|&i| registered[i]).collect(),
        Some(_) => {
            let mut seen = vec![false; registered.len()];
            for &p in &point_ids {
                for &(img, _) in &points[p].track {
                    if registered[img] {
                        seen[img] = true;
                    }
                }
            }
            (0..registered.len()).filter(|&i| seen[i]).collect()
        }
    };
    let image_pos: HashMap<usize, usize> = image_ids
        .iter()
        .enumerate()
        .map(|(compact, &orig)| (orig, compact))
        .collect();
    if image_ids.len() < 2 {
        return;
    }

    let mut camera_of_image: Vec<usize> = Vec::new();
    let mut camera_list: Vec<CameraModel> = Vec::new();
    let mut camera_id_list: Vec<u32> = Vec::new();
    let mut camera_index_of: HashMap<u32, usize> = HashMap::new();
    for &orig in &image_ids {
        let cam_id = input.images[orig].camera_id;
        let idx = *camera_index_of.entry(cam_id).or_insert_with(|| {
            camera_list.push(cameras[&cam_id]);
            camera_id_list.push(cam_id);
            camera_list.len() - 1
        });
        camera_of_image.push(idx);
    }

    let ba_poses: Vec<Pose> = image_ids
        .iter()
        .map(|&orig| poses[orig].clone().unwrap())
        .collect();
    let ba_points: Vec<Vector3<f64>> = point_ids.iter().map(|&p| points[p].xyz).collect();
    // Gauge/fixed-pose selection. `Global` fixes only the seed (6 of the 7
    // gauge dof; scale comes from the seed pose sitting at the origin - see
    // the callers). `Local` additionally fixes every image outside the
    // co-visible window, which both keeps the solve small and pins the
    // window rigidly to the already-converged surrounding model.
    let mut fixed_poses: Vec<bool> = match &free_images {
        None => image_ids.iter().map(|&orig| orig == seed_i).collect(),
        Some(free) => image_ids
            .iter()
            .map(|&orig| orig == seed_i || !free.contains(&orig))
            .collect(),
    };
    // Poses the caller pinned (see `ImageInput::pose_fixed`) are fixed in
    // every pass regardless of scope - that is the whole point of supplying
    // measured extrinsics.
    for (slot, &orig) in fixed_poses.iter_mut().zip(&image_ids) {
        if input.images[orig].pose_fixed && input.images[orig].initial_pose.is_some() {
            *slot = true;
        }
    }
    // A local bundle whose window happens to cover *every* image in the
    // problem (early in growth, when little is registered yet) would have no
    // fixed pose at all, leaving the solve gauge-free and singular. Fall back
    // to pinning the lowest-indexed image, matching what the seed does for
    // the global case.
    if !fixed_poses.iter().any(|&f| f) {
        fixed_poses[0] = true;
    }

    let mut observations = Vec::new();
    for (compact_p, &orig_p) in point_ids.iter().enumerate() {
        for &(img_idx, kp_idx) in &points[orig_p].track {
            let Some(&compact_img) = image_pos.get(&img_idx) else {
                continue;
            };
            let (x, y) = keypoint_px(&input.images[img_idx].features, kp_idx);
            observations.push(sfm_ba::Observation {
                image_idx: compact_img,
                point_idx: compact_p,
                x,
                y,
            });
        }
    }
    if observations.is_empty() {
        return;
    }

    // Cameras are optimized *jointly* with poses/points in one linear system
    // (`BaInput::fixed_cameras`), not as a separate alternating pass - see
    // `sfm_ba`'s module docs for why an earlier alternating design measurably
    // failed to correct calibration on real photos (a wrong-but-self-
    // consistent focal length has almost no gradient once poses/points have
    // already converged to fit it; only a joint solve's combined Jacobian can
    // reliably escape that).
    //
    // Refine focal length and distortion but hold the principal point fixed
    // at its initial guess: unlike focal length, it's weakly constrained by
    // ordinary photos, and jointly refining it anyway measurably made
    // calibration *worse* on real test data (see PLAN.md) - matching why
    // COLMAP holds it fixed by default too.
    let fixed_camera_params = sfm_ba::default_fixed_params_mask(&camera_list);
    // The intrinsics-enabled call happens exactly once (see the caller), so
    // it can afford a much larger iteration budget to fully reconverge
    // poses/points around the newly-refined intrinsics; periodic pose/point-
    // only passes during growth stay at the default for speed.
    let max_iterations = match intrinsics {
        // The final authoritative calibration pass happens once per
        // reconstruction, so it can afford to fully reconverge poses/points
        // around the refined intrinsics.
        IntrinsicsMode::FreeGuarded | IntrinsicsMode::FreeBounded => 200,
        // The growth-time passes only need to *track* the intrinsics as the
        // model grows, not converge them - the final pass above does that.
        // Letting these inherit the 200-iteration budget made them chase full
        // convergence at every growth trigger, which on `sceaux_castle` cost
        // more than it saved. 50 is COLMAP's own
        // `ba_global_max_num_iterations` default. It is not a free parameter
        // to tune: too low is actively harmful, because a half-converged
        // focal length is worse than an unrefined one - every pose and point
        // then converges to fit the wrong intermediate value. Measured on
        // `temple_sparse_ring`, a 25-iteration budget produced 1.16px mean
        // reprojection error and an 18% focal error, against 0.18px / 0.82%
        // at 50.
        IntrinsicsMode::Free => 50,
        // Local bundles deliberately keep the full default budget rather
        // than COLMAP's tighter `ba_local_max_num_iterations` of 25. Tried
        // and reverted: 25 bought 0.5s on `temple_ring` but cost
        // `temple_sparse_ring` its calibration (focal error 0.82% -> 6.16%,
        // points 1832 -> 1549). That dataset's self-calibration sits on a
        // knife edge - small changes in the growth trajectory flip it between
        // basins - and accuracy there is worth more than the half second.
        IntrinsicsMode::Fixed => sfm_ba::BaParams::default().max_iterations,
    };
    let ba_params = sfm_ba::BaParams {
        robust_loss: ba_robust_loss,
        max_iterations,
        ..Default::default()
    };

    // A bootstrap-triangulated point's *scale* is only a guess, never
    // independently verified against the reconstruction's established
    // baseline the way an ordinary two-view triangulation between two
    // already-trusted poses is (see `PointWork::from_bootstrap`'s docs) -
    // exclude those observations from the intrinsics-refining pass so they
    // can't bias a *shared* focal length, even under Huber loss (which
    // down-weights an outlier's gradient but never removes it, and can't
    // catch a self-consistent-but-wrong-scale point at all since its
    // reprojection error looks fine). Only applied when doing so leaves
    // every non-fixed image still covered by at least one observation -
    // otherwise that image's pose block would be entirely unconstrained in
    // this pass, so fall back to the full set for safety.
    // An earlier attempt excluded bootstrap-sourced points' observations
    // from this pass entirely (their triangulated *scale* is only a guess,
    // never independently verified against the reconstruction's
    // established baseline - see `PointWork::from_bootstrap`'s docs, still
    // accurate motivation). Reverted: on `temple_sparse_ring`'s long
    // bootstrap-heavy chain, excluding even *some* observations left enough
    // points with only one surviving observation (2 equations for 3
    // unknowns) to destabilize the shared Schur-complement solve outright -
    // reprojection error went from 0.31px to 25.7px, not just "somewhat
    // worse". `filter_and_reoptimize`'s residual-based filtering below is
    // the safer tool for this: it only ever drops an observation whose
    // *current* fit is actually bad, never blind-drops sound ones.
    let build_input = |fixed_cameras: Vec<bool>,
                       obs: &[sfm_ba::Observation],
                       param_mask: &[Vec<bool>],
                       cameras: &[CameraModel]| sfm_ba::BaInput {
        camera_of_image: camera_of_image.clone(),
        cameras: cameras.to_vec(),
        poses: ba_poses.clone(),
        points: ba_points.clone(),
        observations: obs.to_vec(),
        fixed_poses: fixed_poses.clone(),
        fixed_cameras,
        fixed_camera_params: param_mask.to_vec(),
    };

    // Judge the two candidates by *plain* mean reprojection error, not the
    // robust-loss-weighted cost `bundle_adjust` itself optimizes against:
    // Huber/Cauchy loss is deliberately forgiving of outliers mid-iteration,
    // which also makes it a poor judge of whether the final fit is actually
    // good (see `sfm_ba::mean_reprojection_error`'s docs).
    let eval_error = |output: &sfm_ba::BaOutput| -> f64 {
        sfm_ba::mean_reprojection_error(&sfm_ba::BaInput {
            camera_of_image: camera_of_image.clone(),
            cameras: output.cameras.clone(),
            poses: output.poses.clone(),
            points: output.points.clone(),
            observations: observations.clone(),
            fixed_poses: vec![],
            fixed_cameras: vec![],
            fixed_camera_params: vec![],
        })
    };

    // Only the intrinsics-refining pass gets outlier filtering: it's the one
    // place a few badly-triangulated points can actually bias a *shared*
    // parameter (focal length/distortion) rather than just their own
    // already-somewhat-independent pose/point, and periodic in-loop calls
    // during growth favor speed over this extra refinement.
    let run_ba_with = |fixed_cameras: Vec<bool>,
                       param_mask: &[Vec<bool>],
                       cameras: &[CameraModel]|
     -> sfm_ba::BaOutput {
        let input = build_input(fixed_cameras, &observations, param_mask, cameras);
        if matches!(
            intrinsics,
            IntrinsicsMode::FreeGuarded | IntrinsicsMode::FreeBounded
        ) {
            filter_and_reoptimize(input, &ba_params, max_reprojection_error_px)
        } else {
            sfm_ba::bundle_adjust(input, &ba_params)
        }
    };
    // Staging this as COLMAP does - focal alone first (`ba_refine_focal_length`),
    // then distortion (`ba_refine_extra_params`) - was tried and measured
    // *worse* on the `scan` rig: 2.977px against 0.868px for the single
    // combined solve, with the rig no flatter. `focal_only_params_mask` is kept
    // because the reasoning behind it still holds and the mask is the awkward
    // part to rebuild, but the staging is not wired in. See decisions.md.
    let run_ba = |fixed_cameras: Vec<bool>| -> sfm_ba::BaOutput {
        run_ba_with(fixed_cameras, &fixed_camera_params, &camera_list)
    };

    // `Free` is the lightweight growth-time path: let the intrinsics move
    // with the model, skip the filtering and the fixed-vs-free comparison
    // (both belong to the final authoritative pass), and write the result
    // straight out.
    if intrinsics == IntrinsicsMode::Free {
        let free_fixed_cameras: Vec<bool> = {
            let mut per_cam = vec![0usize; camera_list.len()];
            for &c in &camera_of_image {
                per_cam[c] += 1;
            }
            per_cam
                .iter()
                .enumerate()
                .map(|(idx, &n)| {
                    n < MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
                        || input.fixed_cameras.contains(&camera_id_list[idx])
                })
                .collect()
        };
        let out = run_ba(free_fixed_cameras);
        for (compact, &orig) in image_ids.iter().enumerate() {
            poses[orig] = Some(out.poses[compact]);
        }
        for (compact, &orig) in point_ids.iter().enumerate() {
            points[orig].xyz = out.points[compact];
        }
        for (idx, &cam_id) in camera_id_list.iter().enumerate() {
            cameras.insert(cam_id, out.cameras[idx]);
        }
        return;
    }

    let fixed_output = run_ba(vec![true; camera_list.len()]);

    // Self-calibration (recovering intrinsics from image observations alone)
    // needs real diversity in camera motion to be well-conditioned. Two
    // distinct failure modes showed up on real test data with too little of
    // it (see PLAN.md): a narrow-baseline capture can leave the fit *worse*
    // than not refining at all (caught by the error comparison below); and,
    // more insidiously, with very few images a flexible-enough distortion
    // model can *lower* reprojection error by fitting a wildly unphysical
    // coefficient that "explains away" a wrong focal length on the
    // particular points available, rather than genuinely correcting it - a
    // classic sparse-data overfit that a plain error comparison won't catch
    // (its whole point is to look like a better fit). Require a minimum
    // number of genuinely different views per camera before trusting it with
    // any intrinsics refinement at all, on top of the error comparison.
    let mut images_per_camera = vec![0usize; camera_list.len()];
    for &c in &camera_of_image {
        images_per_camera[c] += 1;
    }
    // Observations per camera, as the fallback measure for the final pass.
    let mut obs_per_camera = vec![0usize; camera_list.len()];
    for o in &observations {
        obs_per_camera[camera_of_image[o.image_idx]] += 1;
    }
    let final_pass = intrinsics == IntrinsicsMode::FreeBounded;
    let eligible = |idx: usize| -> bool {
        images_per_camera[idx] >= MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
            || (final_pass && obs_per_camera[idx] >= MIN_OBSERVATIONS_FOR_INTRINSICS)
    };
    let any_camera_eligible = (0..camera_list.len()).any(eligible);

    let output = if allow_intrinsics && any_camera_eligible {
        let free_fixed_cameras: Vec<bool> = (0..camera_list.len())
            .map(|idx| !eligible(idx) || input.fixed_cameras.contains(&camera_id_list[idx]))
            .collect();
        let free_output = run_ba(free_fixed_cameras);
        // Belt-and-suspenders plausibility bound: a real photographic lens's
        // radial/tangential distortion coefficients are essentially never
        // this large, so an optimizer output that produces one is treated as
        // a bad optimum outright, regardless of what its own reprojection
        // error says.
        //
        // 0.5 is a real bound, not a formality: normal (non-fisheye) lenses
        // sit well inside |k1| < 0.3, and wide-angle rarely passes -0.4. The
        // bound was 2.0, which was loose enough that nothing ever tripped it
        // - `temple_sparse_ring` was observed converging to k1 = -0.575 with
        // a focal length 9.9% off the known truth, *winning* the reprojection
        // comparison below because the two errors cancel in the reprojection
        // but not in the calibration. That is exactly the "self-consistent
        // but wrong" optimum this bound exists to catch, so it is set where
        // it can actually catch one.
        const MAX_PLAUSIBLE_DISTORTION: f64 = 0.5;
        let distortion_is_plausible = free_output.cameras.iter().all(|cam| {
            cam.opencv_distortion()
                .iter()
                .all(|&d| d.abs() < MAX_PLAUSIBLE_DISTORTION)
        });
        // On `temple_sparse_ring` specifically (16 images, ~2.1 observations
        // per point on average) this comparison reliably rejects free_output:
        // investigated directly (see decisions.md) by instrumenting this pass
        // and confirming free_err *starts* above fixed_err and gets *worse*,
        // not better, with a much larger iteration budget (200 -> 600 moved
        // focal length further from the true value and raised free_err from
        // 1.07px to 1.23px) - ruling out slow convergence and confirming a
        // genuine local-optimum/identifiability problem: too few independent
        // multi-view constraints on this dataset's short tracks for a shared
        // focal length to be reliably recoverable from image observations
        // alone. This is exactly the failure mode the comparison below exists
        // to catch; falling back to `fixed_output` here is the correct,
        // working behavior, not a bug to chase further.
        // A focal that has run far from its starting guess is not a
        // calibration, it is a diverged solve. COLMAP bounds the same thing
        // with `min/max_focal_length_ratio` (0.1x to 10x); this is tighter
        // because the initial guess here is a field-of-view heuristic rather
        // than EXIF, and a real lens is not 10x off it.
        let focal_is_plausible =
            free_output
                .cameras
                .iter()
                .zip(&camera_list)
                .all(|(refined, initial)| {
                    let (f_new, _) = refined.focal_lengths();
                    let (f_old, _) = initial.focal_lengths();
                    f_old > 0.0
                        && f_new.is_finite()
                        && (f_new / f_old) > MIN_FOCAL_RATIO
                        && (f_new / f_old) < MAX_FOCAL_RATIO
                });
        let plausible = distortion_is_plausible && focal_is_plausible;
        if std::env::var("SFMTORY_DEBUG_INTRINSICS").is_ok() {
            eprintln!(
                "[intrinsics] pass={:?} distortion_ok={distortion_is_plausible} focal_ok={focal_is_plausible}",
                intrinsics
            );
            for (refined, initial) in free_output.cameras.iter().zip(&camera_list) {
                eprintln!(
                    "   {} f {:.1} -> {:.1}  (ratio {:.3})  distortion {:?}",
                    refined.name(),
                    initial.focal_lengths().0,
                    refined.focal_lengths().0,
                    refined.focal_lengths().0 / initial.focal_lengths().0,
                    refined
                        .opencv_distortion()
                        .iter()
                        .map(|d| (d * 1000.0).round() / 1000.0)
                        .collect::<Vec<_>>()
                );
            }
            eprintln!(
                "   free_err {:.4} vs fixed_err {:.4}",
                eval_error(&free_output),
                eval_error(&fixed_output)
            );
        }
        if intrinsics == IntrinsicsMode::FreeBounded {
            // The caller judges; this pass only refuses the implausible.
            if plausible {
                free_output
            } else {
                fixed_output
            }
        } else if plausible && eval_error(&free_output) < eval_error(&fixed_output) {
            free_output
        } else {
            fixed_output
        }
    } else {
        fixed_output
    };

    for (compact, &orig) in image_ids.iter().enumerate() {
        poses[orig] = Some(output.poses[compact].clone());
    }
    for (compact, &orig) in point_ids.iter().enumerate() {
        points[orig].xyz = output.points[compact];
    }
    for (idx, &cam_id) in camera_id_list.iter().enumerate() {
        cameras.insert(cam_id, output.cameras[idx]);
    }
}

/// Iteratively drops observations whose reprojection error (under the
/// current fit) exceeds `threshold_px`, re-running bundle adjustment on the
/// survivors each round - COLMAP's `FilterObservations`/`FilterPoints`
/// pattern. Huber loss down-weights an outlier's gradient contribution but
/// never removes it entirely; a handful of badly-triangulated points (most
/// commonly from a lower-confidence bootstrap registration, see
/// `try_bootstrap_bridge_image`) can still measurably drag a *shared*
/// camera's intrinsics away from the true value even under robust loss,
/// which is exactly what widened `sceaux_castle`'s focal length error once
/// bootstrap-registered images 10/11 started contributing to the same
/// joint solve - see PLAN.md.
fn filter_and_reoptimize(
    input: sfm_ba::BaInput,
    ba_params: &sfm_ba::BaParams,
    threshold_px: f64,
) -> sfm_ba::BaOutput {
    let camera_of_image = input.camera_of_image.clone();
    let fixed_poses = input.fixed_poses.clone();
    let fixed_cameras = input.fixed_cameras.clone();
    let fixed_camera_params = input.fixed_camera_params.clone();
    let mut observations = input.observations.clone();

    let mut output = sfm_ba::bundle_adjust(input, ba_params);

    const FILTER_ROUNDS: usize = 2;
    const MIN_OBSERVATIONS: usize = 12;
    for _ in 0..FILTER_ROUNDS {
        let before = observations.len();
        observations.retain(|obs| {
            let pose = &output.poses[obs.image_idx];
            let cam = &output.cameras[camera_of_image[obs.image_idx]];
            let point = &output.points[obs.point_idx];
            let pc = pose.transform_point(point);
            if pc.z <= 1e-9 {
                return false;
            }
            let (px, py) = cam.project(&pc);
            ((px - obs.x).powi(2) + (py - obs.y).powi(2)).sqrt() <= threshold_px
        });
        if observations.len() == before || observations.len() < MIN_OBSERVATIONS {
            break;
        }
        output = sfm_ba::bundle_adjust(
            sfm_ba::BaInput {
                camera_of_image: camera_of_image.clone(),
                cameras: output.cameras.clone(),
                poses: output.poses.clone(),
                points: output.points.clone(),
                observations: observations.clone(),
                fixed_poses: fixed_poses.clone(),
                fixed_cameras: fixed_cameras.clone(),
                fixed_camera_params: fixed_camera_params.clone(),
            },
            ba_params,
        );
    }
    output
}

fn assemble_reconstruction(
    input: &ReconstructionInput,
    cameras: &HashMap<u32, CameraModel>,
    registered: &[bool],
    poses: &[Option<Pose>],
    points: &[PointWork],
) -> Reconstruction {
    let mut recon = Reconstruction::new();
    // Only cameras actually used by a registered image are meaningful in the
    // output, but including all input cameras verbatim is harmless and simpler.
    // Width/height come from the original input (they never change); the
    // intrinsic model itself comes from `cameras`, which carries whatever
    // `run_bundle_adjustment` refined it to - `input.cameras` would still
    // have the pre-refinement initial guess.
    for (&camera_id, camera) in &input.cameras {
        let mut camera = camera.clone();
        if let Some(&refined) = cameras.get(&camera_id) {
            camera.model = refined;
        }
        recon.cameras.insert(camera_id, camera);
    }

    for (idx, image_input) in input.images.iter().enumerate() {
        if !registered[idx] {
            continue;
        }
        let pose = poses[idx].clone().unwrap();
        let mut image = Image::new_unregistered(
            image_input.image_id,
            image_input.camera_id,
            image_input.name.clone(),
        );
        image.pose = pose;
        image.keypoints = image_input
            .features
            .keypoints
            .iter()
            .map(|kp| (kp.x, kp.y))
            .collect();
        image.point3d_ids = vec![None; image.keypoints.len()];
        recon.images.insert(image_input.image_id, image);
    }

    let mut next_id = 1u64;
    for point in points {
        if point.track.len() < 2 {
            continue;
        }
        let mut track_elements = Vec::new();
        let mut total_error = 0.0;
        let mut n_err = 0usize;
        for &(image_idx, kp_idx) in &point.track {
            if !registered[image_idx] {
                continue;
            }
            let pose = poses[image_idx].as_ref().unwrap();
            let cam = &cameras[&input.images[image_idx].camera_id];
            // Exact pixel-space error via the camera's full projection
            // (including distortion), *not* `to_normalized` +
            // `reprojection_error_normalized`: that pair ignores distortion
            // entirely (fine for triangulation/PnP inputs when `k` is still
            // the initial all-zero guess, but wrong - understating error by
            // however much distortion the final refined model actually has -
            // once bundle adjustment has refined a nonzero `k`).
            let (obs_x, obs_y) = keypoint_px(&input.images[image_idx].features, kp_idx);
            let pc = pose.transform_point(&point.xyz);
            if pc.z > 1e-9 {
                let (px, py) = cam.project(&pc);
                let e = ((px - obs_x).powi(2) + (py - obs_y).powi(2)).sqrt();
                total_error += e;
                n_err += 1;
            }
            track_elements.push(TrackElement {
                image_id: input.images[image_idx].image_id,
                point2d_idx: kp_idx,
            });
        }
        if track_elements.len() < 2 {
            continue;
        }
        let mean_error = if n_err > 0 {
            total_error / n_err as f64
        } else {
            0.0
        };

        let point3d_id = next_id;
        next_id += 1;
        recon.points3d.insert(
            point3d_id,
            Point3D {
                id: point3d_id,
                xyz: point.xyz,
                color: [180, 180, 180],
                error: mean_error,
                track: track_elements.clone(),
            },
        );
        for te in &track_elements {
            if let Some(image) = recon.images.get_mut(&te.image_id) {
                if let Some(slot) = image.point3d_ids.get_mut(te.point2d_idx as usize) {
                    *slot = Some(point3d_id);
                }
            }
        }
    }

    recon
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;
    use sfm_core::{Descriptors, Keypoint};

    fn pinhole(f: f64, w: u32, h: u32) -> Camera {
        Camera {
            camera_id: 1,
            model: CameraModel::SimplePinhole {
                f,
                cx: w as f64 / 2.0,
                cy: h as f64 / 2.0,
            },
            width: w,
            height: h,
        }
    }

    /// True relative pose of camera `j` given camera `i` as the identity
    /// reference frame - i.e. what `sfm-match`'s essential-matrix decomposition
    /// would (up to scale) recover for that pair. Used here to hand
    /// `run_incremental` an exact synthetic view graph without going through
    /// real feature matching.
    fn relative_pose(true_poses: &[Pose], i: usize, j: usize) -> Pose {
        let ri_inv = true_poses[i].rotation.inverse();
        let rotation = true_poses[j].rotation * ri_inv;
        let translation = true_poses[j].translation - rotation * true_poses[i].translation;
        Pose::from_rotation_translation(rotation, translation)
    }

    #[test]
    fn incremental_pipeline_recovers_synthetic_multiview_scene() {
        let cam = pinhole(750.0, 640, 480);
        let true_poses = vec![
            Pose::identity(),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(0.0, 0.12, 0.0),
                Vector3::new(0.6, 0.0, 0.05),
            ),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(0.05, 0.22, -0.02),
                Vector3::new(1.1, 0.08, 0.15),
            ),
            Pose::from_rotation_translation(
                UnitQuaternion::from_euler_angles(-0.03, 0.3, 0.01),
                Vector3::new(1.6, -0.05, 0.3),
            ),
        ];
        let true_points: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new(
                    0.5 * (t * 0.37).sin(),
                    0.4 * (t * 0.53).cos(),
                    4.0 + 0.08 * t,
                )
            })
            .collect();

        let mut images = Vec::new();
        for (idx, pose) in true_poses.iter().enumerate() {
            let mut keypoints = Vec::with_capacity(true_points.len());
            for point in &true_points {
                let (x, y) = cam.model.project(&pose.transform_point(point));
                keypoints.push(Keypoint {
                    x: x as f32,
                    y: y as f32,
                    scale: 1.0,
                    angle: 0.0,
                    response: 1.0,
                });
            }
            let n = keypoints.len();
            images.push(ImageInput {
                image_id: (idx + 1) as u32,
                camera_id: 1,
                name: format!("img{idx}.png"),
                initial_pose: None,
                pose_fixed: false,
                features: FeatureSet {
                    keypoints,
                    descriptors: Descriptors::Float32 {
                        dim: 1,
                        data: vec![0.0; n],
                    },
                },
            });
        }

        // Every pair sees the full point set, which makes seed-pair choice
        // ("most inlier matches") a tie between all 6 pairs; `max_by_key`
        // breaks ties by taking the *last* candidate, so without forcing a
        // winner the seed (and therefore the whole reconstruction's gauge/
        // anchor frame) would land on an arbitrary pair instead of (0, 1) -
        // equally valid, but not what this test's ground-truth comparison
        // assumes. Give every other pair one fewer match so (0, 1) uniquely
        // wins and the reconstruction is anchored at camera 0, matching
        // `true_poses[0] == identity`.
        let mut pairs = Vec::new();
        for i in 0..images.len() {
            for j in (i + 1)..images.len() {
                let count = if (i, j) == (0, 1) {
                    true_points.len()
                } else {
                    true_points.len() - 1
                };
                let matches: Vec<(u32, u32)> = (0..count as u32).map(|k| (k, k)).collect();
                pairs.push(PairInput {
                    i,
                    j,
                    geometry: TwoViewGeometryRecord {
                        pose: relative_pose(&true_poses, i, j),
                        inlier_matches: matches,
                    },
                });
            }
        }

        let mut cameras = HashMap::new();
        cameras.insert(1u32, cam);
        let input = ReconstructionInput {
            images,
            cameras,
            pairs,
            fixed_cameras: Default::default(),
        };

        let recon = run_incremental(&input, &IncrementalParams::default());

        assert_eq!(
            recon.images.len(),
            4,
            "expected all 4 synthetic images to register"
        );
        assert!(
            recon.points3d.len() >= 30,
            "expected most points to survive triangulation, got {}",
            recon.points3d.len()
        );

        // Every registered image's pose should match ground truth (up to the
        // seed-anchored gauge - camera 0 is fixed at identity, and the seed
        // pair's exact relative pose fixes scale too, so no alignment step
        // should be needed for a noiseless synthetic scene).
        for image in recon.images.values() {
            let true_idx = (image.id - 1) as usize;
            let true_pose = &true_poses[true_idx];
            let center_err = (image.pose.camera_center() - true_pose.camera_center()).norm();
            assert!(
                center_err < 0.05,
                "image {} camera center off by {center_err}",
                image.id
            );
            let angle_err = image.pose.rotation.angle_to(&true_pose.rotation);
            assert!(
                angle_err < 0.01,
                "image {} rotation off by {angle_err} rad",
                image.id
            );
        }

        for point in recon.points3d.values() {
            let te = &point.track[0];
            let true_idx = te.point2d_idx as usize;
            let err = (point.xyz - true_points[true_idx]).norm();
            assert!(
                err < 0.05,
                "point {} off by {err}: got {:?} want {:?}",
                point.id,
                point.xyz,
                true_points[true_idx]
            );
        }
    }
}
