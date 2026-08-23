# sfmtory — Advanced Structure-from-Motion Library

A from-scratch, GUI-free, terminal-first SfM/camera-calibration pipeline written in Rust.
Goal: match or beat COLMAP's calibration accuracy while being faster, using modern
(2023-2025) detection/matching/SfM research, with every stage runnable as its own
CLI command and every stage's result recorded to disk.

Design references (methods only, no code/binaries reused — all licenses below are for
the *techniques*, our implementation is original Rust):
- COLMAP (incremental SfM, camera models, `.txt` export format) — BSD-license project.
- GLOMAP / "Global Structure-from-Motion Revisited" (Pan et al., ECCV 2024) — global
  rotation+translation averaging pipeline, order-of-magnitude faster than incremental
  SfM at comparable accuracy. This is our default reconstruction engine.
- LightGlue (Lindenberger et al., ICCV 2023) / SuperPoint / DISK / ALIKED — learned
  detectors+matchers, higher inlier ratio & robustness on wide baselines than SIFT.
- MAGSAC++ / LO-RANSAC — outlier-free-threshold robust two-view estimation.
- Nerfstudio `transforms.json` convention for NeRF-format export.

## 0. Ground rules

- [x] Language: **Rust** (predictable perf, no GC pauses, fearless parallelism via
      `rayon`, single static binary — easy to ship as CLI tools).
- [x] No GUI. Every stage is a `sfm <subcommand>`. Every stage persists its output to
      a project directory so steps can be run independently, resumed, and inspected.
- [x] All dependencies must be commercially-usable (MIT/Apache-2.0/BSD/Public-Domain).
      No GPL, no research-only weights bundled. SIFT is patent-free since 2020 —
      safe to implement natively.
- [x] Sparse only for now (features + 3D point tracks + poses). No dense/MVS stage.
- [x] Every numeric stage must be swappable for a speed/accuracy tradeoff via CLI flags
      (e.g. `--detector sift|akaze|orb|superpoint|disk|aruco`,
      `--pipeline incremental|global`).

## 1. Repo layout (Cargo workspace)

- [x] `crates/sfm-core` — shared types: `Image`, `CameraModel` (PINHOLE, SIMPLE_RADIAL,
      RADIAL, OPENCV, OPENCV_FISHEYE), `Pose` (SE3), `Track`, `Point3D`, config structs,
      error types. (Project sqlite database still TODO — see sfm-features/sfm-match.)
- [x] `crates/sfm-io` — COLMAP text model reader/writer (`cameras.txt`/`images.txt`/
      `points3D.txt`) and NeRF `transforms.json` reader/writer, both with passing
      round-trip tests including the COLMAP<->NeRF axis-convention conversion.
      Image loading (`image` crate) still TODO, needed by sfm-features.
- [x] `crates/sfm-features` — feature extraction. Done: **SIFT** (native, full
      Gaussian/DoG pyramid, subpixel+edge-rejected extrema, orientation histogram,
      128-d trilinear descriptor), **ORB** (native, multi-scale FAST-9 + intensity-
      centroid orientation + steered BRIEF-256), **ArUco-style fiducial markers**
      (native: adaptive threshold, connected components, convex-hull/rotating-
      calipers quad fit, homography bit-sampling, own generated dictionary —
      see module docs in `aruco.rs` for why it isn't OpenCV-dictionary-compatible).
      All three have passing unit tests on synthetic images. Still TODO: AKAZE
      (via the `akaze` crate) and the ONNX-based deep detectors (SuperPoint/
      DISK/ALIKED), deferred since they need externally-sourced model weights
      whose commercial license would need checking per-file before bundling.
- [x] `crates/sfm-match` — descriptor matching (brute-force mutual-NN + Lowe
      ratio for float/SIFT, Hamming ratio for binary/ORB, exact-ID for ArUco
      corners) + geometric verification (calls `sfm-geometry`'s RANSAC-verified
      essential-matrix estimation, gates pairs on `min_inliers`/`min_inlier_ratio`
      so a weak/coincidental match never reaches the reconstruction). Also
      absorbed the `exhaustive`/`sequential` pairing strategies originally
      scoped as a separate `sfm-pairing` crate (§ below) - small enough not to
      warrant their own crate; `vocab-tree`/`spatial`/`aruco` co-visibility
      pairing are still TODO and matter once image counts get large (O(n^2)
      exhaustive pairing doesn't scale). Brute-force matching is O(n1*n2) per
      pair - fine at current default max-features, a kd-tree (`kiddo`) is the
      next speed upgrade if profiling shows it dominating `sfm match`.
      Unit-tested including a full synthetic match+RANSAC+pose-recovery pipeline.
- [x] `crates/sfm-geometry` — two-view geometry, done and unit-tested on synthetic
      data: normalized 8-point F/E estimation + adaptive RANSAC (custom seeded
      xorshift sampler, not the `rand` crate, for reproducible `sfm map` runs) +
      LO refit, E-decomposition relative-pose recovery with cheirality voting,
      linear (6-point DLT) PnP + RANSAC for later image registration, multi-view
      DLT triangulation, triangulation-angle/reprojection-error helpers.
      Deliberately uses the linear 8-point/6-point solvers instead of minimal
      5-point (Nister)/3-point (P3P) solvers - see module docs in `essential.rs`/
      `pnp.rs` for the reasoning (fewer lines to get right, LO refit erases the
      accuracy gap, only RANSAC iteration count differs). Caught and fixed a
      real bug along the way: the textbook Procrustes "closest rotation" SVD
      sign-fix breaks when the recovered projection block's singular values are
      (near-)degenerate, which happens whenever the true rotation is small -
      the exact-scalar-multiple case needs a whole-matrix sign flip instead
      (see the comment in `pnp.rs::pnp_dlt`).
- [x] `crates/sfm-ba` — sparse bundle adjustment: Levenberg-Marquardt with a
      real Schur-complement reduction (eliminate points, solve a dense reduced
      camera system via `nalgebra` Cholesky, back-substitute for points),
      Huber/Cauchy robust loss (IRLS reweighting), numerical (finite-difference)
      Jacobians instead of per-camera-model analytic ones (see module docs for
      why - same reasoning as sfm-geometry's linear-solver choices), and
      explicit gauge-fixing via `fixed_poses` (anchoring at least one camera is
      mathematically required - reprojection error is invariant under a
      similarity transform of the whole scene, so unanchored BA "converges"
      correctly to an arbitrarily transformed copy of the right answer, not a
      wrong one - this is expected behavior, not a bug, and is now documented
      + handled). Unit-tested end-to-end on a synthetic perturbed 2-camera
      scene. Caught and fixed a real sign bug in the Schur back-substitution
      formula for point updates along the way. Camera intrinsics are now
      optimized **jointly** with poses/points in the same reduced system
      (`BaInput::fixed_cameras`/`fixed_camera_params`) rather than held fixed
      - validated against real COLMAP output on real photos (§ real-data-
      testing entries below); reached this design after an initial
      alternating-pass approach measurably failed to correct calibration.
      Rig constraints and analytic Jacobians remain future work; `faer`-
      backed sparse (rather than dense) Cholesky is a scale optimization to
      revisit once image counts push past the low thousands.
- [~] `crates/sfm-reconstruction` — the two SfM engines:
      - [x] **incremental** (implemented, currently the *only* working
        pipeline despite "global" being the eventually-intended default):
        seed-pair selection (max inlier count) → triangulate seed → loop:
        next-best-view selection by 2D-3D correspondence count → PnP+RANSAC
        registration → triangulate the newly-registered image's fresh
        correspondences → periodic Schur-complement bundle adjustment (every
        `run_ba_every_n_images`) → final BA. Unit-tested end-to-end on a
        synthetic 4-camera/40-point scene (poses recovered to <0.05 units,
        points to <0.05 units, zero alignment step needed). Wired into
        `sfm map --pipeline incremental` and produces real COLMAP-text output
        from real images via the full `extract → match → map → export` CLI
        chain (manually verified).
      - [ ] **global** (GLOMAP-style, not yet implemented): view graph →
        outlier-pruned rotation averaging (chordal L1 + IRLS + cycle-
        consistency triplet filtering) → global translation/position
        averaging → track building (union-find + angle/error filtering) →
        global BA → track filtering/retriangulation → final BA. This is what
        should eventually become the speed-optimized default per PLAN's
        original goals; `sfm map --pipeline global` currently rejects with a
        clear "not implemented" error.
      - Known simplifications (see module docs in
        `crates/sfm-reconstruction/src/lib.rs`): seed-pair choice is "most
        inlier matches" not COLMAP's fuller well-conditioned-ness score; no
        iterative re-triangulation/track-merging pass after registration;
        point color is a fixed gray placeholder, not sampled from source
        images.
- [x] `crates/sfm-cli` — the `sfm` binary; `project new`/`extract`/`match`/
      `map`/`export`/`run` are fully wired to real implementations (not
      stubs); `refine` and `eval` remain typed stubs (see §3). Uses `tracing`
      for structured logs and writes a JSON report per stage under
      `<project>/logs/`. `indicatif` progress bars not yet added.
- [ ] `crates/sfm-eval` — reconstruction comparison/quality tool: Umeyama alignment
      of two reconstructions (ours vs. COLMAP, or vs. ground truth), reports camera
      position/rotation error, reprojection error stats, #registered images, runtime.

## 2. Project directory layout (created by `sfm project new`)

```
my_project/
  sfm.toml              # project config (camera groups, chosen detector/matcher/pipeline)
  images/                # (symlink or path reference to source images)
  database.sqlite        # keypoints, descriptors, pairs, matches, two-view geometries
  sparse/0/              # cameras.txt, images.txt, points3D.txt  (+ our own checkpoint.json)
  export/                # transforms.json / other exports land here
  logs/                  # <stage>.json timing + metrics per run, append-only
```

## 3. CLI commands (each independently runnable, each records a `logs/<stage>_<ts>.json`)

- [x] `sfm project new <dir> --images <path>` — scaffold project (sfm.toml, sparse/0,
      export/, logs/). EXIF ingestion still TODO.
- [x] `sfm extract --project <dir> --detector sift|orb|aruco [--max-features N]`
      — real implementation: rayon-parallel per-image detection, camera-by-
      dimensions grouping (reused across re-runs, not duplicated), features
      stored as bincode BLOBs in `database.sqlite`, logged to `logs/extract_*.json`.
      Smoke-tested end-to-end on generated images. `akaze`/`superpoint`/`disk`
      correctly reject with a clear "not implemented" error until sfm-features
      grows them.
- [x] `sfm match --project <dir> --pairing exhaustive|sequential --matcher mnn-ratio
      [--window N]` — real implementation: rayon-parallel match+verify over the
      pair list, verified two-view geometries (pose + inlier matches) stored in
      `database.sqlite`, logged to `logs/match_*.json`. Smoke-tested end-to-end
      on a synthetic overlapping 3-image scene (all 3 pairs correctly verified).
      `vocab-tree`/`spatial`/`aruco` pairing and `lightglue` matching correctly
      reject with a clear "not implemented" error for now.
- [~] `sfm map --project <dir> --pipeline global|incremental` — `incremental` is
      fully wired (real implementation, writes `sparse/0/`, tested end-to-end
      via the CLI on real images); `global` correctly rejects as
      not-implemented for now. Note: the CLI's *default* is still `global`
      per the original PLAN, so `--pipeline incremental` must currently be
      passed explicitly - flip the default once `global` exists.
- [ ] `sfm refine --project <dir> [--robust-loss huber|cauchy] [--refine-intrinsics]`
      — standalone global bundle-adjustment / re-triangulation pass on an existing
      model. Not wired yet, but `sfm-ba` (the hard part) already exists and is
      tested - this should mostly be plumbing: load `sparse/0` via `sfm-io`,
      convert to `sfm_ba::BaInput`, run, write back.
- [x] `sfm export --project <dir> --format colmap-text|nerf-transforms [--out <path>]`
      (tested end-to-end: reads `sparse/0`, writes either format, logs result;
      `--out` now optional, defaulting to the project's `export/` directory).
- [ ] `sfm eval --ours <sparse/0> --baseline <colmap sparse/0> [--gt <path>]` —
      accuracy/speed comparison report (this is how we prove "better than
      COLMAP"). Currently a stub that loads and prints basic stats
      (image/point counts, mean reprojection error) for `--ours` and
      `--baseline` independently; the actual Umeyama-alignment pose-error
      comparison is not implemented yet.
- [x] `sfm run --project <dir> ...` — convenience wrapper chaining
      `extract → match → map → export`, fully wired (was previously calling
      only the first three stages despite claiming to chain through export -
      now actually does). Verified end-to-end via the CLI on real images.

## 4. Camera / calibration correctness

- [x] Support shared intrinsics across images from the same physical camera
      (`CAMERA_ID` grouping, same as COLMAP) — implemented in `sfm extract`
      (images grouped by pixel dimensions into a shared camera, reused across
      re-runs) and respected throughout matching/reconstruction/BA.
- [x] Support multiple distinct cameras in one project (multi-camera rig or mixed
      sources), each with its own `CAMERA_ID` — the data model and `sfm-ba`
      already support this (`camera_of_image` mapping); not yet exercised by
      an actual multi-camera test scene.
- [ ] Optional rigid-rig constraint: fixed relative pose between cameras in a physical
      rig, softly constrained in BA (stretch goal, flagged in code as `--rig-config`).
- [x] Distortion models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV,
      OPENCV_FISHEYE — implemented in `sfm-core::camera` (project/params/name
      round-trip), matching COLMAP naming so outputs are drop-in compatible.

## 5. Outlier / noise control (this is what should beat COLMAP)

- [x] Lowe ratio test + mutual-NN cross-check at match time (`sfm-match`).
- [x] RANSAC (adaptive, seeded/reproducible) two-view geometric verification with
      an inlier-set LO refit; pairs below `min_inliers`/`min_inlier_ratio` are
      rejected outright (`sfm-match::VerificationParams`). Not yet MAGSAC-style
      (threshold-free scoring) - still a fixed-threshold RANSAC, see
      `sfm-geometry::essential` module docs for the minimal-solver tradeoff.
- [ ] Rotation-cycle (triplet) consistency check to prune bad relative rotations before
      rotation averaging — removes edges classic incremental SfM has no equivalent
      cheap check for. Blocked on the global pipeline existing.
- [x] Track quality filtering: minimum triangulation angle, max reprojection error
      (`sfm-reconstruction::triangulate_pair_matches`) before a point is ever
      created; min track length (>=2) enforced at final assembly.
- [ ] Iterative re-triangulation + re-filtering after each BA pass (COLMAP does this;
      we do it with tighter default thresholds tuned during eval, see §7).

## 6. Performance

- [x] `rayon` data-parallelism: per-image feature extraction (`sfm extract`),
      per-pair matching + geometric verification (`sfm match`), per-observation
      BA Jacobian computation (`sfm-ba`) — all implemented. Per-track
      triangulation in `sfm-reconstruction` is still sequential (small enough
      relative to matching/BA that it hasn't been a priority; easy rayon win
      later).
- [ ] `faer` for sparse Schur-complement linear algebra in BA. Currently a
      **dense** reduced camera system via `nalgebra` Cholesky instead (see
      `sfm-ba` module docs) - correct and fine up to the low thousands of
      images, `faer` sparse solving is the scale upgrade beyond that.
- [x] f32 for descriptors/matching (`sfm-core::Descriptors`), f64 for BA normal
      equations and all geometry (`sfm-geometry`, `sfm-ba`).
- [ ] Global pipeline as the speed default (order-of-magnitude faster than incremental
      per GLOMAP benchmarks) with incremental available via flag for hard scenes.
      Blocked on the global pipeline being implemented at all (§1).

### GPU: optional, auto-detected, everywhere it actually pays off

- [ ] `--gpu` flag on `sfm extract` and `sfm match` (already present in the CLI
      skeleton, §3) selects GPU execution when a compatible device is present and
      falls back to CPU automatically with a logged warning if it isn't — never a
      hard failure just because `--gpu` was passed on a CPU-only machine.
- [ ] **Detectors**: classical detectors (SIFT/AKAZE/ORB/ArUco) are CPU algorithms
      by nature and stay CPU + `rayon`-parallel across images — there's no
      commercially-free, portable GPU path for them worth the complexity. The
      learned detectors (SuperPoint/DISK/ALIKED) run through `ort` (ONNX Runtime)
      and pick up whatever execution provider is available: CUDA/TensorRT on
      Linux+NVIDIA, DirectML on Windows, CoreML on macOS, else CPU EP.
- [ ] **Matching**: classical mutual-NN + ratio-test matching (kd-tree via `kiddo`)
      stays CPU — it's already fast enough that GPU transfer overhead would net
      negative at typical (thousands-of-images) scale. LightGlue runs through the
      same `ort` execution-provider selection as the learned detectors, and is
      where GPU actually matters (it's the most expensive per-pair op when enabled).
- [ ] **Bundle adjustment stays CPU-only by design, not by omission**: the
      Schur-complement sparse solve is inherently sequential/data-dependent
      (variable sparsity pattern per scene) and doesn't map cleanly onto GPU
      without a research-grade rewrite; COLMAP/Ceres and GLOMAP's own BA are CPU
      too. `rayon` + `faer` SIMD covers this instead.
- [ ] Runtime GPU capability probe lives in `sfm-features`/`sfm-match` (query `ort`
      available execution providers at startup); `sfm extract`/`sfm match --gpu`
      report which provider was actually selected in their `logs/<stage>_*.json`.

## 7. Validation ("prove it's better than COLMAP")

- [x] Acquire public benchmark datasets with known ground truth (`data/sceaux_castle`,
      `data/temple_sparse_ring`, `data/temple_ring` — see `data/README.md`).
- [x] Run real COLMAP (via `pycolmap`) on the same images as a baseline, for all three.
- [x] Run `sfm run`/`sfm map` on the same images and compare focal length error,
      mean reprojection error, and registered-image count against COLMAP.
- [x] `sceaux_castle`: matches COLMAP's registered-image count (11/11); beats it on
      reprojection error; focal length error still slightly behind COLMAP's.
- [x] `temple_sparse_ring`: beats COLMAP's registered-image count (16/16 vs 13/16);
      focal length error still behind COLMAP's.
- [x] `temple_ring` (47 images, added to test at larger scale): matches COLMAP's
      registered-image count (47/47) and beats it on reprojection error and focal
      length error - a clean sweep on all three metrics.
- [ ] Close the remaining focal-length-accuracy gap on `sceaux_castle` and
      `temple_sparse_ring` (see `decisions.md`'s "Known open gaps").
- [ ] `sfm eval`'s ground-truth-alignment accuracy path (Umeyama alignment vs.
      `templeSR_par.txt`) — not yet wired up as a CLI command.
- [ ] Tune default thresholds further and document final results in `BENCHMARKS.md`.

## 8. Optional GUI (future)

- [ ] Viewer/front-end on top of the existing CLI pipeline (pose/point-cloud
      inspection, driving extract/match/map/export without hand-typed commands).
      CLI + on-disk project format remain the primary, scriptable interface either way.

## 9. Status log

Titles only — see `decisions.md` for durable rationale and git history for details.

- [x] Workspace scaffolded; `sfm-core`, `sfm-io` implemented with tests.
- [x] End-to-end pipeline working (`sfm-features`, `sfm-geometry`, `sfm-match`, `sfm-ba`,
      `sfm-reconstruction`); `sfm extract`/`match`/`map`/`run` functional, not stubs.
- [x] `README.md` and `docs/USER_GUIDE.md` added.
- [x] Real test datasets added (`data/sceaux_castle`, `data/temple_sparse_ring`,
      `data/temple_ring`); first real-data pipeline run and bug-fix pass.
- [x] Real COLMAP head-to-head comparison via `pycolmap`; joint intrinsics
      (focal length + distortion) self-calibration implemented in `sfm-ba`.
- [x] Registration-count parity/improvement on both test datasets (deterministic
      RANSAC sampling, degenerate-PnP-sample rejection, nonlinear PnP polish,
      connected-component-aware multi-seed selection, bridge-image bootstrap
      for chain-shaped match graphs, SIFT 2x upsampling for small images).
- [x] Bundle-adjustment outlier filtering (`sceaux_castle` focal error 5.4% → 3.1%)
      and analytic (not numerical) Jacobians for every camera model, validated against
      real Ceres Solver output (twice, including on `temple_ring`) then shipped
      dependency-free each time (`temple_sparse_ring` focal error 6.3% → 3.7%).
- [x] Validated at larger scale (`temple_ring`, 47 images): matches/beats COLMAP on
      all three metrics, including focal length error (1.2% vs. COLMAP's 2.0%).
- [ ] Close remaining focal-length-accuracy gap vs. COLMAP on `sceaux_castle` and
      `temple_sparse_ring` (3.1% vs. 2.3%, 3.7% vs. 0.02%).
- [ ] Proper P3P minimal solver (would tighten PnP further, esp. near-planar scenes).
- [ ] `sfm map --pipeline global` (GLOMAP-style rotation/translation averaging).
- [ ] `sfm eval` ground-truth-alignment accuracy path.
- [ ] AKAZE + ONNX deep detectors (SuperPoint/DISK/ALIKED) + LightGlue matching.
- [ ] Vocab-tree/spatial/ArUco-covisibility pairing for scaling past ~300 images.
