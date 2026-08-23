# Architecture & design decisions

Durable rationale behind non-obvious choices in sfmtory's design, organized by
topic rather than chronologically. Each entry says *what* was chosen and *why*;
the fine-grained "how" lives in the relevant module's own doc comments — this
file is the index, not a duplicate.

## Pipeline structure

- **Incremental SfM (seed pair → triangulate → PnP-register next-best-view →
  repeat → periodic bundle adjustment) is the only implemented pipeline.** A
  GLOMAP-style global (rotation/translation-averaging) pipeline is designed
  for but not yet built — see PLAN.md's checklist. Incremental is slower at
  scale but simpler to get correct, and correctness against real data turned
  out to need multiple non-obvious fixes (below) even for this simpler design.
- **Camera intrinsics are optimized *jointly* with poses/points in one linear
  system** (`sfm-ba`'s `BlockRef`/Schur-complement generalization), not via an
  alternating block-coordinate scheme. An earlier alternating design measurably
  failed to correct a wrong-but-self-consistent focal length: once poses/points
  converge to fit it, the intrinsics-only gradient nearly vanishes. Only a
  joint solve's combined Jacobian reliably escapes that local optimum (see
  `sfm-ba`'s `joint_optimization_recovers_shared_camera_focal_length` test).
- **Intrinsics only refine once, in the final bundle adjustment call** —
  never during incremental growth. Refining mid-growth means every
  subsequently-registered image gets triangulated against a still-changing
  calibration, which measurably failed to reconverge within a reasonable
  iteration budget. Periodic in-loop BA passes during growth keep intrinsics
  fixed and just stabilize poses/points.
- **Principal point (`cx`/`cy`) stays fixed; focal length + distortion
  refine.** Jointly refining the principal point sent it wildly off (weakly
  constrained by ordinary photos) — matches COLMAP's own default behavior.
  See `sfm-ba::intrinsics::default_fixed_params_mask`.
- **Self-calibration is gated, not unconditional**: a camera needs >=5
  images before its intrinsics are refined at all, the free-vs-fixed result
  is picked by *plain* mean reprojection error (not the robust/Huber-weighted
  cost the optimizer itself minimizes, which is deliberately forgiving of
  outliers and a poor judge of overall fit), and an implausible resulting
  distortion coefficient (`|d| >= 2.0`) is rejected outright regardless of
  what the error comparison says. All three guard against real observed
  failure modes: too few images making self-calibration ill-conditioned, and
  sparse-data overfitting where a flexible distortion model "explains away" a
  wrong focal length rather than correcting it.
- **Outlier filtering runs only in the intrinsics-refining pass**
  (`filter_and_reoptimize`): iteratively drop observations whose reprojection
  error exceeds threshold, re-run BA on the survivors. Huber loss down-weights
  an outlier's gradient contribution but never removes it entirely, which
  measurably let a handful of badly-triangulated points bias the *shared*
  focal length. This is a residual-based filter — it only ever drops an
  observation whose current fit is actually bad. A related idea (excluding
  bootstrap-sourced points' observations from this pass outright, since their
  triangulated scale is a guess rather than independently verified) was tried
  and reverted: on `temple_sparse_ring`'s bootstrap-heavy chain it left enough
  points with only one surviving observation to destabilize the shared
  Schur-complement solve (reprojection error jumped to 25px). Residual-based
  filtering doesn't have that failure mode since it only removes what's
  already numerically bad.

## Feature extraction

- **SIFT includes Lowe's original 2x pre-upsampling, but only below
  `UPSAMPLE_MAX_MIN_DIM` (1600px min dimension).** Upsampling finds
  substantially more low-contrast/small keypoints — critical on already-small
  images (`temple_sparse_ring`'s 640x480 photos extracted only ~400 features
  without it, starving match density). But it's a 4x compute/memory blowup,
  and applying it unconditionally to large photos (`sceaux_castle`'s
  2832x2128 originals) OOM-crashed the dev environment when 11 of them
  extracted in parallel — large images have no shortage of resolution to
  begin with, so the size gate costs nothing there.

## Two-view geometry & PnP

- **Essential-matrix RANSAC uses the linear 8-point solver, not a minimal
  5-point one**; PnP RANSAC uses linear 6-point DLT, not a minimal 3-point
  P3P solver. Both trade some RANSAC sample-efficiency for a much simpler,
  easier-to-verify implementation — acceptable since both do a full inlier-set
  local-optimization refit afterward.
- **PnP RANSAC rejects near-coplanar minimal samples before fitting.** Linear
  PnP-DLT solves for an 11-DOF projective camera; a coplanar 3D point sample
  only constrains an 8-DOF homography, so the system genuinely loses rank
  rather than merely becoming ill-conditioned (Hartley & Zisserman's
  "critical configurations"). Undetected, this silently returns a pose only a
  handful of points agree with by chance. Scored via the ratio of the
  sample's covariance's smallest to largest eigenvalue.
- **PnP RANSAC's result is polished by a dedicated Gauss-Newton reprojection
  refinement** (`refine_pose_gauss_newton`) after the existing linear DLT
  refit, alternated with re-classifying inliers. Linear DLT minimizes
  algebraic error, not true reprojection error, and stays sensitive to input
  point noise — this was the single highest-impact registration fix found:
  one previously-9/212-inlier image jumped to 186/210 after nonlinear polish,
  confirming the pose was directionally fine but algebraically noisy.
- **RANSAC sampling order must be deterministic input, even with a fixed
  internal seed.** `sfm map` was silently non-deterministic (7/11, 8/11,
  9/11 registered across identical repeated runs) because next-best-view
  correspondence gathering built its candidate list via a `HashMap`, whose
  iteration order is randomized per-process — a fixed RANSAC seed samples by
  *index* into that order, so a random order means a random actual subset
  each run. Fixed by sorting correspondences by keypoint index before RANSAC
  ever sees them.

## Seed & registration-graph structure

- **Seed-pair candidates are restricted to the match graph's largest
  connected component** before any quality scoring runs. A pair can score
  well on match density/triangulation-angle while being otherwise isolated
  (zero other pairs touch either image) — no seed placed outside a component
  can ever grow into it, silently capping registration regardless of any
  other fix.
- **Seed selection tries several top-ranked candidates and keeps whichever
  actually grows into the most registered images**, rather than trusting a
  single static quality heuristic. No proxy score reliably predicts how far a
  *given* seed will grow once bridge/bootstrap paths and PnP successes/
  failures compound downstream — mirrors COLMAP's own incremental mapper,
  which has the identical multi-candidate fallback. See `grow_from_seed` /
  `GrowthResult` in `sfm-reconstruction`.
- **A "bridge image" bootstrap handles chain-shaped match graphs.** Ordinary
  PnP registration needs a *triangulated* 3D point shared with a registered
  neighbor, which requires either a triangle in the match graph or incidental
  keypoint-level overlap. A graph that's mostly a spanning tree (as
  `temple_sparse_ring`'s turned out to be, past one well-connected quad) has
  "bridge" images with exactly one registered neighbor and *zero* shared
  triangulated points — not a thin correspondence set, an empty one,
  unfixable by triangulating more of the rest of the scene first. The
  fallback composes an absolute pose directly from the bridge pair's cached,
  already-verified two-view relative pose (exact rotation; translation
  rescaled from the essential matrix's arbitrary unit baseline to the
  reconstruction's real scale via a multiplier sweep around the median
  inter-camera baseline), refines it against any existing correspondences if
  3+ exist, and validates by how much of the pair's own raw matches then
  triangulate cleanly. Runs a fixed-intrinsics bundle adjustment immediately
  after every bootstrap step, since these poses carry materially more
  uncertainty than an ordinary RANSAC/GN-verified PnP registration. See
  `try_bootstrap_bridge_image`, `compose_pose_via_neighbor`,
  `find_bridge_candidate`.

## Analytic Jacobians ("miniceres")

- **`sfm-ba` uses exact analytic Jacobians for every `CameraModel` variant**
  (`SimplePinhole`, `Pinhole`, `SimpleRadial`, `Radial`, `OpenCV`,
  `OpenCVFisheye`), not finite-difference ones - hand-derived via the chain
  rule through each model's own `project()` formula, and verified against
  the (still-kept, test-only) numerical Jacobians in a dedicated unit test
  per model (`analytic_matches_numerical_for_*`). This started narrower and
  widened once the first version proved out: `SIMPLE_RADIAL` alone closed a
  real focal-length-accuracy gap on `temple_sparse_ring`'s harder self-
  calibration problem (fewer images, more extreme viewing angles than
  `sceaux_castle`) - central-difference Jacobians there measurably converged
  to a worse local optimum (6.3% focal length error) than Ceres Solver's
  autodiff-based solver did on the identical input (3.7%), while matching
  Ceres almost exactly on `sceaux_castle`'s easier problem. Confirmed
  empirically, not just theorized: a temporary Ceres-backed bundle-
  adjustment backend (`sfm-ba-ceres`, a small C++ shim + Rust FFI wrapper,
  built and removed twice now - see below) A/B tested this, isolated to
  *only* the final intrinsics-refining pass so periodic in-loop calls
  wouldn't confound the comparison with a different growth trajectory.
  Deriving the remaining five models' Jacobians the same way immediately
  caught a real bug in the process: an OpenCV tangential-distortion
  coefficient (6·p1·yp, not 4·p1·yp, in `d(yd)/d(yp)`) that the per-model
  numerical cross-check test flagged on the first run - exactly the kind of
  mistake hand-derived calculus is prone to, and exactly why every model got
  its own correctness test rather than trusting the derivation by symmetry
  with `SIMPLE_RADIAL`'s already-validated formula.
- **Validated on a third, larger real dataset** (`data/temple_ring`, 47
  images - see "Third dataset" below) in addition to the original two: both
  the native solver and Ceres converge to essentially identical results
  there (1504.8 vs. 1505.0 focal length, both 0.298px reprojection error) -
  a well-conditioned self-calibration problem where both solvers agree, as
  expected. On `temple_sparse_ring`'s harder problem, re-running the same
  Ceres A/B comparison in a later session found Ceres itself landing in the
  *worse* basin consistently across repeated runs (768/k=0, the implausible-
  distortion-rejected fallback) - including with Ceres forced to a single
  thread, ruling out floating-point-reduction-order nondeterminism as the
  cause - while the native analytic-Jacobian solver stayed at the same
  stable 3.7%-error result every time, matching the *best* result Ceres had
  ever produced on this input rather than its later, worse one. Not fully
  root-caused (a Ceres/library version difference between the two sessions
  is the leading guess), but the practical takeaway holds either way: the
  native solver is at least as accurate as Ceres on every dataset tried, and
  more reliably deterministic than Ceres was on the one genuinely
  ill-conditioned case.
- **A Ceres Solver-backed bundle-adjustment backend was built twice, used to
  validate the above both times, and deliberately removed both times**
  (`sfm-ba-ceres`, scoped to SIMPLE_RADIAL only, selected via a
  `SFM_BA_BACKEND=ceres` env var during each investigation). This is
  explicitly a diagnostic tool, not a permanent architecture choice - keeping
  it would make Ceres (a dynamic system library) a de facto runtime
  dependency, breaking the project's "single static binary" deployment goal,
  for no remaining benefit once native quality matches or beats it. If
  revisiting self-calibration accuracy again (a new camera model, a new hard
  dataset), rebuilding the same diagnostic (compare native vs. a mature
  reference solver on the *same* real data, isolated to just the pass under
  test) is a fast, reliable way to tell whether a discrepancy is a
  precision/implementation issue (fixable natively, as it was here) or
  something more fundamental (e.g. a genuinely hard-to-avoid local optimum).
  Requires `apt install libceres-dev libeigen3-dev libgflags-dev
  libgoogle-glog-dev` (prebuilt packages, no source build) to reconstruct.

## Third dataset: `temple_ring`

- **Added `data/temple_ring`** (47 images, Middlebury's `templeRing` - the
  same physical rig/object/intrinsics as `temple_sparse_ring`, just a real
  walk-around ring instead of that dataset's sparse two-latitude-band
  sampling) specifically to test at a larger image count than the original
  two datasets (11 and 16 images) support. Requested to confirm sfmtory's
  fixes hold up with more images, not just avoid regressing on the original
  two. Result: **47/47 registered (beats COLMAP's 47/47 - tied), 0.298px
  mean reprojection error (beats COLMAP's 0.319px), 1.2% focal length error
  (beats COLMAP's 2.0%)** - a clean sweep on all three metrics, and by a
  wider margin than on the original two datasets. More images giving a
  better-conditioned self-calibration problem is the likely reason both
  solvers' accuracy improves here relative to the sparser sets.

## Known open gaps

- **Focal length error is still somewhat worse than COLMAP's on both test
  datasets** (`sceaux_castle`: ~3.1% vs COLMAP's 2.32%; `temple_sparse_ring`:
  ~3.7% vs COLMAP's ~0.02%), despite reprojection error and registered-image
  count now matching or beating COLMAP on both, and despite closing a
  meaningful chunk of the gap via outlier filtering (`sceaux_castle`: was
  5.4%) and analytic Jacobians (`temple_sparse_ring`: was 6.3%, see above).
  The remaining gap - especially `temple_sparse_ring`'s, where COLMAP's
  0.02% is near-perfect - looks like it may be specific to *which* points
  and observations feed the final joint solve (COLMAP's own feature
  extraction/matching may simply produce a cleaner, better-conditioned set
  for this particular hard dataset) rather than anything left to fix in the
  optimizer itself. Not yet root-caused further. Tried and reverted:
  excluding bootstrap-sourced observations from the intrinsics pass outright
  (too fragile on temple's chain-heavy graph, see the bundle-adjustment
  entry above). Tried with no effect on `sceaux_castle`: more BA iterations,
  more filter rounds, a stricter bootstrap-acceptance threshold (which also
  regressed temple's registration count 16/16 → 12/16, reverted).
- **One two-view pair in `temple_sparse_ring` (images 8 and 9) has a
  near-degenerate cached relative pose**: zero of its 39 verified inlier
  matches triangulate to a well-conditioned point even using the pair's own
  geometry standalone (no composition involved), across a scale sweep
  spanning four orders of magnitude — ruling out a scale/composition bug.
  Most likely a near-pure-rotation (translation-starved) motion between
  those two specific viewpoints, unrecoverable from epipolar geometry alone.
  Observed before SIFT upsampling was added (above), which changed the whole
  match graph (21 verified pairs instead of 15, new triangles at 7-8-9 and
  3-4-14) enough that 16/16 now registers without hitting this pair's
  degeneracy at all — not re-confirmed whether the (8, 9) pair itself is
  still degenerate under the denser features, just no longer load-bearing.
