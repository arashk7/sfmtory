//! Camera-model selection by rebuilding the reconstruction.
//!
//! `modelselect` scores candidate models against the *existing* structure. That
//! is cheap and it is the right tool for choosing how many distortion
//! coefficients a lens wants, but it is structurally blind to the question a
//! user actually asks first - "is this lens even a rectilinear camera?" - because
//! the points it scores against were triangulated by the incumbent model. On a
//! four-camera rig with two fisheye lenses it recommended `SIMPLE_RADIAL` for
//! all four, and the two fisheyes came out 8% of the rig's own spacing off the
//! plane they are physically bolted to.
//!
//! The only comparison that can see this runs the reconstruction again. A wrong
//! projection family cannot hide from re-triangulation: it has to place the
//! camera at the wrong depth to make its periphery fit, and the resulting
//! structure is measurably worse. That is expensive (one rebuild per
//! candidate), so it is a deliberate, opt-in step rather than something `map`
//! does on every run.
//!
//! Two things make the search affordable enough to be worth offering:
//!
//! - Candidates are tried *uniformly* first, then refined one camera at a
//!   time. A joint search over families for `n` cameras is `|families|^n`; a
//!   uniform pass followed by coordinate descent is `|families| * (1 + n)`
//!   and, on the rigs tested here, lands in the same place.
//! - The focal is only swept for cameras that cannot refine their own
//!   intrinsics. When a camera has enough images, bundle adjustment moves the
//!   focal itself and sweeping it is wasted work. When it has one image, the
//!   focal cannot be recovered at all - and since a fisheye's focal for the
//!   same field of view is roughly a third of a rectilinear one's, seeding
//!   every family from the incumbent's focal would judge each family on the
//!   wrong focal rather than on the family.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use sfm_core::{Camera, CameraModel};
use sfm_reconstruction::{IncrementalParams, ReconstructionInput};

/// Families worth rebuilding for, simplest first. The order is the parsimony
/// order: ties and near-ties resolve to the earliest entry.
pub const FAMILIES: &[&str] = &[
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "RADIAL3",
    "OPENCV",
    "OPENCV_FISHEYE",
];

/// Focal multipliers tried for a camera whose intrinsics cannot be refined.
///
/// Spans a fisheye's focal through a rectilinear one's: an equidistant fisheye
/// covering the same field of view as a rectilinear lens has a focal around
/// `(fov/2) / tan(fov/2)` times as large, which is well under half for the wide
/// lenses this matters for. Coarse on purpose - this is here to stop the family
/// comparison being decided by the focal, not to calibrate.
pub const FOCAL_SCALES: &[f64] = &[0.3, 0.4, 0.55, 0.75, 1.0, 1.3];

/// How much better a more complex family has to be before it is preferred.
///
/// Reprojection error falls monotonically with parameter count on the data it
/// was fitted to, so "lowest error wins" always picks the most complex family
/// offered and would happily fit eight distortion coefficients to a pinhole
/// webcam. A candidate only displaces a simpler one by beating it by this
/// margin; conversely a simpler family wins if it comes within this margin of
/// the best. 10% is well below the gap a genuinely wrong family produces - the
/// fisheye rig above was 70% worse as `SIMPLE_RADIAL` - and well above the
/// fraction an extra coefficient buys when the family is already right.
pub const PARSIMONY_MARGIN: f64 = 0.10;

/// How much of the best candidate's structure a candidate must keep to be
/// considered at all.
///
/// Mean reprojection error is an average over *surviving* points, so the
/// cheapest way to a small number is to triangulate almost nothing. Without
/// this bound the search did exactly that: against a 311-point baseline at
/// 3.141px it happily recommended configurations holding 60 points at 0.301px
/// and called them a twelvefold improvement. They are not better calibrations,
/// they are emptier ones. Completeness is therefore ranked above accuracy, the
/// same way registering an image is - a point that failed to triangulate is a
/// hole in the result, not a favourable rounding.
pub const POINT_RETENTION: f64 = 0.9;

/// A candidate's measured outcome. Deliberately the same three numbers `map`
/// prints, so a recommendation can be checked by re-running `map` by hand.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    pub registered: usize,
    pub points: usize,
    pub mean_error: f64,
}

impl Score {
    fn failed() -> Self {
        Score {
            registered: 0,
            points: 0,
            mean_error: f64::INFINITY,
        }
    }

    fn is_usable(&self) -> bool {
        self.registered >= 2 && self.points > 0 && self.mean_error.is_finite()
    }
}

/// One camera's chosen model and the focal it was seeded with.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub model: &'static str,
    pub focal: f64,
    /// Whether the focal was swept, i.e. whether this camera had too few
    /// images for bundle adjustment to move it.
    pub focal_swept: bool,
}

pub type Assignment = BTreeMap<u32, Choice>;

/// Builds a camera with `model`'s parameter layout at focal `focal`, keeping the
/// principal point and starting every distortion term at zero.
pub fn seed_camera(cam: &Camera, model: &str, focal: f64) -> Option<Camera> {
    let (cx, cy) = cam.model.principal_point();
    let params: Vec<f64> = match model {
        "SIMPLE_PINHOLE" => vec![focal, cx, cy],
        "PINHOLE" => vec![focal, focal, cx, cy],
        "SIMPLE_RADIAL" => vec![focal, cx, cy, 0.0],
        "RADIAL" => vec![focal, cx, cy, 0.0, 0.0],
        "RADIAL3" => vec![focal, cx, cy, 0.0, 0.0, 0.0],
        "OPENCV" | "OPENCV_FISHEYE" => vec![focal, focal, cx, cy, 0.0, 0.0, 0.0, 0.0],
        _ => return None,
    };
    Some(Camera {
        camera_id: cam.camera_id,
        model: CameraModel::from_name_and_params(model, &params)?,
        width: cam.width,
        height: cam.height,
    })
}

fn num_params(model: &str) -> usize {
    match model {
        "SIMPLE_PINHOLE" => 3,
        "PINHOLE" => 4,
        "SIMPLE_RADIAL" => 4,
        "RADIAL" => 5,
        "RADIAL3" => 6,
        "OPENCV" | "OPENCV_FISHEYE" => 8,
        _ => usize::MAX,
    }
}

/// Strips descriptor payloads while keeping every descriptor's *variant*.
///
/// The reconstruction never reads a descriptor's contents - matching already
/// happened - it only asks what kind they are, to tell an identity-proven
/// fiducial correspondence from a nearest-neighbour one (see
/// `correspondence_is_proven`). Keypoints and inlier matches it does read, and
/// those stay.
///
/// This matters because the search makes one copy of the input per concurrent
/// trial. On a SIFT dataset the descriptors are one to two orders of magnitude
/// larger than everything else put together, so carrying them into every trial
/// would turn an affordable search into an out-of-memory risk for no effect on
/// the result. Verified by rebuilding with and without: byte-identical models.
pub fn lighten(input: &ReconstructionInput) -> ReconstructionInput {
    let images = input
        .images
        .iter()
        .map(|im| {
            let mut features = im.features.clone();
            features.descriptors = match &features.descriptors {
                sfm_core::Descriptors::Float32 { dim, .. } => sfm_core::Descriptors::Float32 {
                    dim: *dim,
                    data: Vec::new(),
                },
                sfm_core::Descriptors::Binary {
                    bytes_per_descriptor,
                    ..
                } => sfm_core::Descriptors::Binary {
                    bytes_per_descriptor: *bytes_per_descriptor,
                    data: Vec::new(),
                },
                // Marker corners *are* the identity the matcher proved, and
                // they are 12 bytes per keypoint - nothing to save, and the
                // variant carries meaning the reconstruction acts on.
                other => other.clone(),
            };
            sfm_reconstruction::ImageInput {
                image_id: im.image_id,
                camera_id: im.camera_id,
                name: im.name.clone(),
                features,
                initial_pose: im.initial_pose,
                pose_fixed: im.pose_fixed,
            }
        })
        .collect();
    ReconstructionInput {
        images,
        cameras: input.cameras.clone(),
        pairs: input.pairs.clone(),
        fixed_cameras: input.fixed_cameras.clone(),
    }
}

/// Runs one reconstruction with `assignment` applied and scores the result.
pub fn trial(
    input: &ReconstructionInput,
    assignment: &Assignment,
    params: &IncrementalParams,
) -> Score {
    let mut cameras = input.cameras.clone();
    for (id, choice) in assignment {
        let Some(base) = input.cameras.get(id) else {
            continue;
        };
        match seed_camera(base, choice.model, choice.focal) {
            Some(c) => {
                cameras.insert(*id, c);
            }
            None => return Score::failed(),
        }
    }
    // `run_incremental` takes the input by reference and reads its cameras from
    // it, so a trial needs its own input. Only the cameras differ, but there is
    // no way to say that through this signature - hence the copy. `lighten`
    // keeps it cheap.
    let trial_input = ReconstructionInput {
        images: input.images.clone(),
        cameras,
        pairs: input.pairs.clone(),
        fixed_cameras: input.fixed_cameras.clone(),
    };
    let recon = sfm_reconstruction::run_incremental(&trial_input, params);
    if recon.images.len() < 2 || recon.points3d.is_empty() {
        return Score::failed();
    }
    Score {
        registered: recon.images.len(),
        points: recon.points3d.len(),
        mean_error: recon.mean_reprojection_error(),
    }
}

/// Picks the winner from a set of scored candidates.
///
/// Registering more images beats any error improvement - an unregistered image
/// is a hole in the result, not a slightly worse number - and below that, the
/// simplest family within `PARSIMONY_MARGIN` of the best error wins.
fn pick(scored: &[(Choice, Score)]) -> Option<&(Choice, Score)> {
    let usable: Vec<usize> = (0..scored.len())
        .filter(|&i| scored[i].1.is_usable())
        .collect();
    let most_registered = usable.iter().map(|&i| scored[i].1.registered).max()?;
    // Completeness before accuracy, twice over: never drop an image, then never
    // collapse the structure. See `POINT_RETENTION`.
    let complete: Vec<usize> = usable
        .into_iter()
        .filter(|&i| scored[i].1.registered == most_registered)
        .collect();
    let most_points = complete.iter().map(|&i| scored[i].1.points).max()?;
    let floor = (most_points as f64 * POINT_RETENTION) as usize;
    let dense: Vec<usize> = complete
        .into_iter()
        .filter(|&i| scored[i].1.points >= floor)
        .collect();
    let best_error = dense
        .iter()
        .map(|&i| scored[i].1.mean_error)
        .fold(f64::INFINITY, f64::min);
    let cutoff = best_error * (1.0 + PARSIMONY_MARGIN);
    // `FAMILIES` is in parsimony order, so the earliest-listed family that
    // clears the cutoff is the simplest adequate one.
    let winner = dense
        .into_iter()
        .filter(|&i| scored[i].1.mean_error <= cutoff)
        .min_by_key(|&i| {
            let model = scored[i].0.model;
            (
                num_params(model),
                FAMILIES.iter().position(|f| *f == model).unwrap_or(99),
            )
        })?;
    Some(&scored[winner])
}

/// Whether `new` should displace `incumbent` as the running best.
///
/// Same order as `pick`: never register fewer images, never collapse the
/// structure, then reduce the error.
fn improves(new: &Score, incumbent: &Score) -> bool {
    if !new.is_usable() {
        return false;
    }
    if !incumbent.is_usable() {
        return true;
    }
    new.registered >= incumbent.registered
        && (new.points as f64) >= incumbent.points as f64 * POINT_RETENTION
        && new.mean_error < incumbent.mean_error
}

/// One row of the search, kept for reporting so the user can see what was tried
/// rather than only what was chosen.
#[derive(Debug, Clone)]
pub struct Tried {
    /// `None` for the uniform pass, which assigns every camera the same family.
    pub camera_id: Option<u32>,
    pub choice: Choice,
    pub score: Score,
}

pub struct Outcome {
    pub assignment: Assignment,
    pub baseline: Score,
    pub best: Score,
    pub tried: Vec<Tried>,
}

/// Cameras with fewer images than this cannot have their focal refined by
/// bundle adjustment, so the search has to supply it. Mirrors the
/// reconstruction's own gate rather than guessing.
fn images_per_camera(input: &ReconstructionInput) -> BTreeMap<u32, usize> {
    let mut n = BTreeMap::new();
    for im in &input.images {
        *n.entry(im.camera_id).or_insert(0) += 1;
    }
    n
}

fn candidates_for(cam: &Camera, sweep_focal: bool) -> Vec<Choice> {
    let (fx, _) = cam.model.focal_lengths();
    let scales: &[f64] = if sweep_focal { FOCAL_SCALES } else { &[1.0] };
    FAMILIES
        .iter()
        .flat_map(|m| {
            scales.iter().map(move |s| Choice {
                model: m,
                focal: fx * s,
                focal_swept: sweep_focal,
            })
        })
        .collect()
}

/// Searches for the camera models that rebuild best.
///
/// `progress` is called with `(done, total)` after each rebuild, because the
/// search takes minutes and silence for minutes reads as a hang.
pub fn search(
    input: &ReconstructionInput,
    params: &IncrementalParams,
    mut progress: impl FnMut(usize, usize),
) -> Outcome {
    let counts = images_per_camera(input);
    let camera_ids: Vec<u32> = input.cameras.keys().copied().collect();
    let sweep: BTreeSet<u32> = camera_ids
        .iter()
        .copied()
        .filter(|id| {
            counts.get(id).copied().unwrap_or(0)
                < sfm_reconstruction::MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS
        })
        .collect();

    let light = lighten(input);
    let input = &light;

    let mut tried = Vec::new();
    let baseline = trial(input, &Assignment::new(), params);

    // Pass 1: every camera gets the same family, which finds the right answer
    // outright on a single-lens dataset and gives coordinate descent a sane
    // starting point on a mixed rig.
    let reference = camera_ids
        .first()
        .and_then(|id| input.cameras.get(id))
        .cloned();
    let Some(reference) = reference else {
        return Outcome {
            assignment: Assignment::new(),
            baseline,
            best: baseline,
            tried,
        };
    };
    let any_sweep = !sweep.is_empty();
    let uniform = candidates_for(&reference, any_sweep);
    let total = uniform.len() + camera_ids.len() * uniform.len();
    let mut done = 0usize;

    let uniform_scored: Vec<(Choice, Score)> = uniform
        .par_iter()
        .map(|c| {
            let assignment: Assignment = camera_ids
                .iter()
                .map(|id| {
                    let f = input
                        .cameras
                        .get(id)
                        .map(|cam| cam.model.focal_lengths().0)
                        .unwrap_or(c.focal);
                    // Scale each camera's *own* focal, so cameras that started
                    // at different focals stay at different focals.
                    let scale = c.focal / reference.model.focal_lengths().0;
                    (
                        *id,
                        Choice {
                            model: c.model,
                            focal: f * scale,
                            focal_swept: sweep.contains(id),
                        },
                    )
                })
                .collect();
            (c.clone(), trial(input, &assignment, params))
        })
        .collect();
    done += uniform_scored.len();
    progress(done, total);

    let mut best_uniform = pick(&uniform_scored).cloned();
    for (c, s) in &uniform_scored {
        tried.push(Tried {
            camera_id: None,
            choice: c.clone(),
            score: *s,
        });
    }

    let mut assignment: Assignment = match &best_uniform {
        Some((c, _)) => camera_ids
            .iter()
            .map(|id| {
                let f = input
                    .cameras
                    .get(id)
                    .map(|cam| cam.model.focal_lengths().0)
                    .unwrap_or(c.focal);
                let scale = c.focal / reference.model.focal_lengths().0;
                (
                    *id,
                    Choice {
                        model: c.model,
                        focal: f * scale,
                        focal_swept: sweep.contains(id),
                    },
                )
            })
            .collect(),
        None => Assignment::new(),
    };
    let mut best = best_uniform
        .take()
        .map(|(_, s)| s)
        .unwrap_or(Score::failed());

    // Pass 2: one camera at a time, everything else held at the incumbent. This
    // is what separates a mixed rig - two rectilinear lenses and two fisheyes -
    // which no uniform assignment can express.
    for id in &camera_ids {
        let Some(cam) = input.cameras.get(id) else {
            continue;
        };
        let cands = candidates_for(cam, sweep.contains(id));
        let scored: Vec<(Choice, Score)> = cands
            .par_iter()
            .map(|c| {
                let mut a = assignment.clone();
                a.insert(*id, c.clone());
                (c.clone(), trial(input, &a, params))
            })
            .collect();
        done += scored.len();
        progress(done, total);

        let winner = pick(&scored).cloned();
        for (c, s) in &scored {
            tried.push(Tried {
                camera_id: Some(*id),
                choice: c.clone(),
                score: *s,
            });
        }
        if let Some((c, s)) = winner {
            if improves(&s, &best) {
                assignment.insert(*id, c);
                best = s;
            }
        }
    }

    Outcome {
        assignment,
        baseline,
        best,
        tried,
    }
}

/// The `[[cameras]]` block that reproduces `assignment`, ready to paste into
/// `sfm.toml`.
pub fn as_camera_toml(
    assignment: &Assignment,
    globs: &BTreeMap<u32, String>,
    pin_focal: bool,
) -> String {
    let mut out = String::new();
    for (id, c) in assignment {
        let glob = globs
            .get(id)
            .cloned()
            .unwrap_or_else(|| format!("*cam{id:03}*"));
        out.push_str("[[cameras]]\n");
        out.push_str(&format!("name = \"cam{id}\"\n"));
        out.push_str(&format!("images = {glob:?}\n"));
        out.push_str(&format!("model = {:?}\n", c.model));
        if pin_focal && c.focal_swept {
            // Only worth writing when the search had to supply the focal: a
            // camera with enough images refines its own, and pinning a swept
            // value there would freeze a coarse guess over a fitted one.
            out.push_str(&format!("# focal from the model search, not calibrated\nparams = [{:.1}, 0.0, 0.0]  # f, cx, cy - fill in the principal point\n", c.focal));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(f: f64) -> Camera {
        Camera {
            camera_id: 1,
            model: CameraModel::SimpleRadial {
                f,
                cx: 100.0,
                cy: 50.0,
                k: 0.1,
            },
            width: 200,
            height: 100,
        }
    }

    fn choice(model: &'static str) -> Choice {
        Choice {
            model,
            focal: 100.0,
            focal_swept: false,
        }
    }

    fn score(reg: usize, pts: usize, err: f64) -> Score {
        Score {
            registered: reg,
            points: pts,
            mean_error: err,
        }
    }

    #[test]
    fn seeding_keeps_principal_point_and_zeroes_distortion() {
        let c = seed_camera(&cam(500.0), "OPENCV_FISHEYE", 180.0).unwrap();
        assert_eq!(c.model.name(), "OPENCV_FISHEYE");
        assert_eq!(c.model.principal_point(), (100.0, 50.0));
        assert_eq!(c.model.focal_lengths(), (180.0, 180.0));
        assert!(c.model.params()[4..].iter().all(|v| *v == 0.0));
        // Image size must survive - a reseeded camera that forgets its size
        // silently changes what "distance from the principal point" means.
        assert_eq!((c.width, c.height), (200, 100));
    }

    #[test]
    fn parsimony_prefers_the_simpler_model_within_the_margin() {
        // OPENCV_FISHEYE is 5% better than SIMPLE_RADIAL - inside the 10%
        // margin, so the four-parameter model should win.
        let scored = vec![
            (choice("SIMPLE_RADIAL"), score(10, 500, 1.00)),
            (choice("OPENCV_FISHEYE"), score(10, 520, 0.95)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "SIMPLE_RADIAL");
    }

    #[test]
    // 3.141 here is scan3's measured reprojection error, not an approximation
    // of pi; the real numbers are the point of the test.
    #[allow(clippy::approx_constant)]
    fn a_clearly_wrong_family_loses() {
        // The scan3 case: the wide model is 40% better, far outside the margin.
        let scored = vec![
            (choice("SIMPLE_RADIAL"), score(4, 300, 3.141)),
            (choice("OPENCV_FISHEYE"), score(4, 306, 1.84)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "OPENCV_FISHEYE");
    }

    #[test]
    fn registering_more_images_outranks_a_lower_error() {
        // A model that drops an image is not "better" for having a tidier
        // error over the images it kept - that is survivorship bias.
        let scored = vec![
            (choice("SIMPLE_RADIAL"), score(9, 400, 0.50)),
            (choice("RADIAL3"), score(10, 500, 2.00)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "RADIAL3");
    }

    #[test]
    fn failed_rebuilds_are_ignored() {
        let scored = vec![
            (choice("SIMPLE_PINHOLE"), Score::failed()),
            (choice("RADIAL"), score(5, 100, 2.0)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "RADIAL");
        let all_bad = vec![(choice("RADIAL"), Score::failed())];
        assert!(pick(&all_bad).is_none());
    }

    #[test]
    fn focal_scales_span_a_fisheye_through_a_rectilinear_lens() {
        // A camera seeded at a rectilinear focal must be able to reach the
        // roughly one-third focal an equidistant fisheye needs for the same
        // field of view, or the family comparison is decided by the focal.
        assert!(FOCAL_SCALES.iter().any(|s| *s <= 0.35));
        assert!(FOCAL_SCALES.iter().any(|s| *s >= 1.0));
    }
    #[test]
    fn an_emptier_reconstruction_is_not_a_better_one() {
        // The bug this guards: against a 311-point baseline the search
        // recommended a configuration holding 60 points because its average
        // over those 60 was small. Completeness outranks accuracy.
        let scored = vec![
            (choice("SIMPLE_RADIAL"), score(4, 306, 1.84)),
            (choice("SIMPLE_PINHOLE"), score(4, 60, 0.30)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "SIMPLE_RADIAL");
    }

    #[test]
    fn a_small_loss_of_points_is_tolerated() {
        // Point counts wobble between rebuilds; only a collapse disqualifies.
        let scored = vec![
            (choice("SIMPLE_RADIAL"), score(4, 300, 2.00)),
            (choice("OPENCV_FISHEYE"), score(4, 290, 1.00)),
        ];
        assert_eq!(pick(&scored).unwrap().0.model, "OPENCV_FISHEYE");
    }

    #[test]
    fn improves_refuses_to_trade_structure_for_error() {
        let incumbent = score(4, 300, 2.0);
        assert!(!improves(&score(4, 100, 0.1), &incumbent));
        assert!(improves(&score(4, 295, 1.5), &incumbent));
        assert!(!improves(&score(3, 300, 0.5), &incumbent));
        assert!(improves(&score(4, 300, 1.9), &Score::failed()));
    }
}
