# sfmtory

[![CI](https://github.com/arashk7/sfmtory/actions/workflows/ci.yml/badge.svg)](https://github.com/arashk7/sfmtory/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)

A from-scratch, CLI-first Structure-from-Motion / camera-calibration pipeline written in
Rust. Point it at a folder of images and get camera intrinsics, camera poses, and a sparse
3D point cloud — exportable as a COLMAP text model or a NeRF-style `transforms.json`.

Every pipeline stage (feature extraction, matching, reconstruction, export) is its own
CLI subcommand with its own on-disk result, so you can run them one at a time, inspect
intermediate state in an ordinary SQLite browser, and resume from any stage. A GUI is
planned as an optional layer on top of this pipeline — see [Roadmap](#roadmap).

> **Status: early, but real.** The `extract → match → map → export` chain works
> end-to-end today and is unit-tested at every layer (41 tests across 8 crates).
> **Benchmarked against both real COLMAP and real GLOMAP on real photos, across three
> datasets** (11, 16, and 47 images). sfmtory registers every image on all three
> (matching or beating both), and **leads both on reprojection error and focal-length
> accuracy on all three**. Against GLOMAP on the largest dataset it wins on *both*
> axes — ~2x faster and better on every accuracy metric; against COLMAP it wins every
> accuracy metric and trails ~8% on map-stage time. See
> [Status vs. COLMAP and GLOMAP](#status-vs-colmap-and-glomap) for the full table and
> the honest remaining gaps. An optional GPU-capable learned detector (**DISK**, via
> ONNX Runtime) is also implemented — see [GPU support](#gpu-support).
> [PLAN.md](PLAN.md) tracks what's implemented and what's stubbed;
> [decisions.md](decisions.md) has the design rationale and full investigation
> write-ups.

## Contents

- [Why](#why)
- [Features](#features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Status vs. COLMAP and GLOMAP](#status-vs-colmap-and-glomap)
- [Viewer (`sfmtory gui`)](#viewer-sfmtory-gui)
- [Estimating intrinsics (`init-cam`)](#estimating-intrinsics-init-cam)
- [Camera setup](#camera-setup)
- [Calibrating from ArUco markers](#calibrating-from-aruco-markers)
- [Evaluating a reconstruction](#evaluating-a-reconstruction)
- [GPU support](#gpu-support)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why

Most sparse-SfM tooling is either a GUI application (COLMAP) or a research codebase built
around Python/PyTorch. sfmtory is neither: a dependency-light, single static binary, with
a pipeline designed to be scripted and automated rather than clicked through.

## Features

- **Feature detection**: native SIFT (full scale-space, no external CV library), native
  ORB, a custom square fiducial-marker ("ArUco-style") detector for rig calibration with
  high-confidence correspondences, and **DISK** — a learned detector run through ONNX
  Runtime, optionally GPU-accelerated (`--detector disk --gpu`), with weights downloaded
  on first use rather than bundled. See [GPU support](#gpu-support). AKAZE and a
  LightGlue matcher are planned but not implemented yet.
- **Matching**: mutual-nearest-neighbor + Lowe's ratio test, RANSAC-verified two-view
  geometry (normalized 8-point + adaptive RANSAC + local optimization), with weak/false
  pairs rejected outright rather than fed downstream.
- **Pairing**: exhaustive, sequential-window, or `vocab-tree` retrieval — a
  hierarchical visual-word vocabulary trained on your own images, so large sets don't
  pay the O(n²) cost of matching every pair.
- **Multi-camera and calibration priors**: declare cameras explicitly when several
  physical cameras share a resolution, and supply known intrinsics and/or extrinsics
  as initialization or as hard constraints — see [Camera setup](#camera-setup).
- **Reconstruction**: incremental SfM — seed-pair triangulation, PnP-based image
  registration, and Schur-complement bundle adjustment, all from an original
  implementation (no COLMAP/OpenCV code reused).
- **Camera models**: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV,
  OPENCV_FISHEYE — the same names and parameter order COLMAP uses, so output is drop-in
  compatible with COLMAP-reading tools.
- **Export**: COLMAP text format (`cameras.txt`/`images.txt`/`points3D.txt`) and NeRF's
  `transforms.json` convention (correct OpenCV↔OpenGL axis conversion included).
- **Multi-camera aware**: images are grouped into shared cameras by resolution, so
  calibration pools observations across every photo from the same physical camera rather
  than treating each image as its own camera.
- Every stage's output and timing is logged to `<project>/logs/<stage>_<timestamp>.json`.
- Deterministic: RANSAC uses a seeded PRNG, not the system RNG, so re-running the same
  images gives the same reconstruction.
- Commercial-friendly dependencies throughout (MIT/Apache-2.0/BSD) — SIFT's patent expired
  in 2020, so this includes SIFT.

## Quick start

Requires a recent stable Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/arashk7/sfmtory.git
cd sfmtory
./install.sh                 # builds release and installs `sfmtory`
./install.sh --uninstall     # to remove it again
```

`install.sh` puts the binary in `/usr/local/bin` when run as root and
`~/.local/bin` otherwise; `--prefix <dir>` overrides that.

Point it at a dataset and run the four stages. The project directory defaults
to the current directory, and no setup file is needed — a folder with an
`images/` directory in it is already a valid project:

```bash
cd my_dataset          # contains images/
sfmtory feature
sfmtory match --pairing exhaustive
sfmtory map --pipeline incremental
sfmtory export --format colmap-text          # or --format nerf-transforms
```

Each stage is a separate process that writes its output under `cache/`:

```text
my_dataset/
  images/
  cache/
    project.sqlite       # shared working store
    feature/             # report.json, corners.csv, aruco_params.toml
    match/               # report.json
    map/sparse/0/        # the reconstruction, COLMAP text format
  export/                # default export destination (--out to override)
```

### Input layouts

All three are detected automatically — captures are optional, and `.jpg`,
`.jpeg` and `.png` are all accepted:

```text
images/capture_000/cam000/image.jpg    # captures x cameras
images/cam000/image.jpg                # cameras only
images/cam000_image.jpg                # flat, optional camNNN_ prefix
```

Camera ids follow the numbers in your directory names (`cam007` → camera 7)
and fall back to sorted order for non-numeric names like `left/`, `right/`.

### Fiducial markers

```bash
sfmtory feature --detector aruco                    # sensible defaults
sfmtory feature --detector aruco --find-params      # tune for this dataset
sfmtory feature --detector aruco --merge-multicaps  # fixed-camera rigs
```

`--find-params` sweeps the ArUco threshold parameters and contrast/gamma
preprocessing against your own images and saves the winner to
`cache/feature/aruco_params.toml`, which later runs pick up automatically.
This matters because mistuned fiducial detection fails *silently and totally* —
on a low-contrast test capture the defaults find zero markers and the search
recovers all of them.

`--merge-multicaps` is for a rig of **fixed** cameras photographing a scene
that changes between captures (moving a marker board and re-shooting). Since
the cameras never move, every capture's observations belong to the same pose,
so merging them per camera turns N sparse captures into one well-constrained
view. On a 3-capture × 4-camera test rig this is the difference between
**0 of 66 pairs verifying and 6 of 6** — without it each capture is below the
inlier floor and there is no reconstruction at all.

Every marker corner is identified as `capture_camera_image_aruco_corner` and
listed with its pixel position in `cache/feature/corners.csv`. The capture is
part of a corner's identity, so a marker you physically moved between captures
is correctly treated as a *different* 3D point rather than matched to itself.

## Architecture

A Cargo workspace, one crate per pipeline concern:

| Crate | Responsibility |
|---|---|
| `sfm-core` | Camera models, poses, the sparse reconstruction data model |
| `sfm-io` | COLMAP text and NeRF `transforms.json` readers/writers |
| `sfm-features` | SIFT / ORB / ArUco-style / DISK (ONNX, GPU-optional) detectors |
| `sfm-geometry` | Two-view geometry, PnP, triangulation, RANSAC |
| `sfm-match` | Descriptor matching + geometric verification |
| `sfm-ba` | Bundle adjustment (Levenberg-Marquardt, Schur complement) |
| `sfm-reconstruction` | The incremental SfM engine tying the above together |
| `sfm-cli` | The `sfm` binary |

Each crate's module docs explain its own deliberate simplifications (e.g. numerical vs.
analytic Jacobians in `sfm-ba`, linear vs. minimal solvers in `sfm-geometry`) — read those
before assuming a shortcut is a bug.

## Status vs. COLMAP and GLOMAP

The goal is to match or beat both real COLMAP's and real GLOMAP's calibration accuracy
*and* registration completeness. Both (via `pycolmap`, which now bundles GLOMAP's
`global_mapping` after the standalone GLOMAP repo was deprecated upstream) have been run
head-to-head against `sfmtory map --pipeline incremental` on the same real photos for all
three test sets in [`data/`](data/README.md), with independently-verified ground-truth
focal lengths (a `K.txt`/`templeR_par.txt`-style calibration file per dataset, not
COLMAP's own output used as a reference).

| Dataset | Metric | sfmtory | COLMAP | GLOMAP |
|---|---|---|---|---|
| `temple_ring` (47 img) | Registered | 47/47 | 47/47 | 47/47 |
| | Points3D | **10599** | 7629 | 6286 |
| | Reprojection error | **0.283px** | 0.296px | 0.312px |
| | Focal length error | **0.58%** | 2.01% | n/a¹ |
| | Map stage time | 15.8s | **14.5s** | 31.5s |
| `temple_sparse_ring` (16 img) | Registered | **16/16** | 13/16 | 16/16 |
| | Points3D | **1832** | 1242 | 1417 |
| | Reprojection error | **0.181px** | 0.228px | 0.205px |
| | Focal length error | **0.82%** | 3.82% | 1.90% |
| | Map stage time | 4.0s | **0.86s** | 1.61s |
| `sceaux_castle` (11 img) | Registered | 11/11 | 11/11 | 11/11 |
| | Points3D | 7758 | **7927** | 7851 |
| | Reprojection error | **0.412px** | 0.611px | 0.598px |
| | Focal length error | **2.18%** | 2.32% | 2.37% |
| | Map stage time | 6.3s | –² | –² |

¹ GLOMAP's `global_mapping` produced per-image cameras on this dataset, so a
single focal-error figure isn't comparable.
² `pycolmap` exhausts memory on this machine at `sceaux_castle`'s 6MP
resolution; those rows' accuracy figures are from an earlier run on the same
images, and no same-machine timing is available to quote.

**sfmtory leads both real systems on reprojection error and focal-length
accuracy on all three datasets**, and on point count on two of three. Against
GLOMAP on `temple_ring` it is a clean sweep on *both* axes — ~2x faster and
better on every accuracy metric. Against COLMAP it wins every accuracy metric
and trails by ~8% on map-stage wall-clock.

Getting there meant replicating what COLMAP's incremental mapper actually
does, which profiling showed this pipeline had diverged from in four
independent ways — local bundle adjustment after each registration instead of
periodic full-model solves, growth-ratio-triggered global bundles, taking the
first viable seed pair instead of growing all candidates, and refining
intrinsics progressively rather than in one jump at the end. Combined with
solver work (fixed variables removed from the linear system, allocation-free
Schur blocks, correctly-scoped parallelism), `temple_ring`'s map stage went
from ~391s to 15.8s across this work — ~25x. Full write-up, including the
things that were tried and *rejected* with numbers (Nielsen trust-region
damping, warm-starting, wrapping Ceres): [`decisions.md`](decisions.md).

There's no `BENCHMARKS.md` write-up yet and `sfmtory eval`'s automated comparison
logic is still a stub — these numbers were produced by hand against real
COLMAP/GLOMAP output on one machine, so treat them as real, reproducible data
points rather than a comprehensive benchmark suite.

## Viewer (`sfmtory gui`)

```bash
cd my_dataset
sfmtory gui

# Or open straight onto one view, which is usually why you opened it at all:
sfmtory gui --view graph        # scene | image | graph | coverage
```

A viewer and front-end over the same pipeline the CLI runs — every button
shells out to the subcommand you would otherwise type, against the same
project directory, so nothing is reachable here that isn't reachable without
it.

- **3D view** of the sparse points and camera frusta. Drag to orbit,
  scroll to zoom, right-drag (or shift-drag) to pan. Colour points by
  reprojection error, hide everything above a residual cutoff, click a point
  to list the images observing it, and click a camera to select it.
- **Image view** shows the selected photo with its detected keypoints, which
  of them triangulated, and each residual as a vector at adjustable
  exaggeration — the quickest way to separate "detection is fine, matching
  isn't" from the reverse.
- **Match graph** draws images as nodes and verified pairs as edges, colours
  connected components, and marks unregistered images hollow. An incremental
  reconstruction grows from one seed and cannot cross a component boundary,
  so a split graph explains unregistered images that nothing in the logs does.
- **Coverage** plots where a planar calibration target has been seen from —
  tilt band against tilt direction — leaving the orientations you still need
  to shoot visibly empty.
- **Right panel** carries the properties of whatever is selected plus three
  diagnostic sections:
  - *Calibration quality* — per camera, whether self-calibration refined the
    focal, rejected its own result, never considered the camera eligible, or
    was pinned by `sfm.toml`; plus the track-length histogram and per-image
    observation counts that explain the verdict.
  - *Viewing-angle diversity* — for a planar target, the best-fit plane and
    the spread of the axis it is tilted about. Rotating a board about one
    fixed axis is degenerate for single-plane self-calibration however wide
    the tilt range, and this is where that shows up.
  - *Reprojection & outliers* — mean/median/p95/max, and the worst
    observations as a clickable list. A mean hides whether a model is
    uniformly decent or mostly excellent with a few broken points, so the
    panel says which.
- **Left panel** browses the current directory; double-click a folder to make
  it the active project.
- **Top bar** runs `init-cam`, `feature`, `match`, `map`, `export` and `eval`
  with the main options exposed as dropdowns. Output streams into a log pane,
  and the model reloads automatically when a stage finishes.

Residuals shown here are recomputed from the model's own poses and intrinsics,
not read from the stored per-point error — the same reason `sfmtory eval`
recomputes them.

The 3D view is rendered on the CPU rather than through a GPU pipeline: a
sparse model is a few tens of thousands of points and a handful of frusta,
comfortably inside what a straightforward rasteriser handles at interactive
rates, and it keeps the viewer free of shaders and a second rendering path.

The GUI is behind a default-on cargo feature, so headless builds can drop the
windowing stack entirely:

```bash
cargo build --release --no-default-features   # CLI only, no GUI dependencies
```

## Estimating intrinsics (`init-cam`)

Every later stage starts from *some* focal length. Without help that is
`1.2 x max(width, height)` — a ~53° field-of-view rule of thumb that is wrong
for most cameras. Bundle adjustment can refine it, but only when the scene
gives it enough multi-view redundancy; when it doesn't, that guess *is* the
answer you get.

```bash
sfmtory init-cam            # report estimates
sfmtory init-cam --apply    # ...and write the winner into sfm.toml
```

It runs several independent estimators and keeps the best-supported one:

| Estimator | Works when | Confidence |
|---|---|---|
| `exif` | the file kept its camera metadata | High |
| `marker-squares` | fiducial markers are **large in frame** | High/Medium |
| `vanishing-points` | scene has strong perpendicular structure | Medium/Low |
| `fov-heuristic` | always (the fallback) | Heuristic |

Two design rules matter more than any individual method:

- **Abstaining is a valid result.** A confidently wrong focal is worse than an
  obvious heuristic, because it looks like an answer. Each estimator returns
  nothing rather than a number it can't support.
- **Independent agreement is evidence.** When two estimators sharing no
  assumptions land on the same focal, that is reported as corroboration.

Real example (`sceaux_castle`, true focal 2905.88 px):

```text
  -> exif              f =   2753.33 px  [High]
       11/11 image(s) carried a 35mm-equivalent focal length; KODAK Z612 @ 35mm
     fov-heuristic     f =   3398.40 px  [Heuristic]
       1.2 x longest image side (~53 degree horizontal field of view)
```

EXIF lands 5.3% from truth against the heuristic's 16.9%, and applying it
improves the reconstruction: **7696 → 7979 points and 0.409 → 0.378px**. Note
the self-calibrated focal ended slightly further out (2.18% → 2.89%) — on that
dataset the focal is weakly constrained regardless of where it starts, so
treat the structure improvement as the win, not the calibration.

`marker-squares` deserves a note, because it needs no user input: **every ArUco
marker is a square**, which is known target geometry, so each detection is a
Zhang calibration constraint. Physical marker size is *not* needed — size only
fixes overall scale. The catch is that a marker small in the image is nearly
affine under projection, its perspective terms fall to the level of corner
noise, and the constraint carries no signal; the estimator detects this (via
the marker's own vanishing points) and abstains. On a 13-photo set where
markers covered ~5% of frame width, all 169 detections were in that regime. It
needs a board that fills a good part of the frame — which is what a deliberate
calibration capture looks like anyway.

Output lands in `cache/init-cam/` as `intrinsics.json` (all estimates with
reasoning) and `cameras.toml` (a ready-to-paste `[[cameras]]` block).

## Camera setup

By default images are grouped into cameras **by resolution**, which is right for a
single camera and wrong when two *different* physical cameras happen to share one —
they'd be merged into a single shared intrinsics block. Declare them explicitly in
`sfm.toml` when that matters:

```toml
images_dir = "/path/to/images"

[[cameras]]
name   = "left"
images = "left_*"          # glob on the file name; `*` and `?` supported

[[cameras]]
name   = "right"
images = "right_*"
model  = "OPENCV"                                  # default SIMPLE_RADIAL
params = [1200, 1200, 640, 480, -0.1, 0.01, 0, 0]  # known intrinsics
refine = false                                     # pin them exactly
```

Known extrinsics work the same way, as initialization or as hard constraints:

```toml
[[poses]]
image       = "left_0001.png"
quaternion  = [1.0, 0.0, 0.0, 0.0]   # world-to-camera, [w, x, y, z]
translation = [0.0, 0.0, 0.0]        # t = -R * camera_center
fixed       = true                   # omit/false to use as initialization only
```

When two or more images carry a pose, those *are* the starting reconstruction — seed
selection is skipped, there's no scale ambiguity to resolve, and every verified pair
among them is triangulated directly. On `temple_ring`, supplying all 47 poses plus
true intrinsics yields **17256 points at 0.130px**, against 10599 at 0.283px when
solving for everything from scratch. Partial priors work too (4 of 47 poses still
registers 47/47). Details and measurements: [`decisions.md`](decisions.md).

## Calibrating from ArUco markers

A complete run on a fiducial-only dataset — no natural features, just markers.
Works at scale: validated on **200 viewpoints of 150 markers**.

```bash
cd my_dataset                 # contains images/

# 1. Detect markers. Add --find-params once if detection looks thin.
sfmtory feature --detector aruco

# 2. Match. Fiducial correspondences are exact-ID, so pair them exhaustively.
sfmtory match --pairing exhaustive

# 3. Reconstruct.
sfmtory map --pipeline incremental

# 4. Check the result against a known focal length.
sfmtory eval --gt-focal 1150
```

Measured on the 200-view synthetic scene (`cargo run --release -p sfm-cli
--example gen_aruco_scene3d -- images 200 1150 1280 960 150`):

| Stage | Result |
|---|---|
| `feature` | 11808 corners across 200 images |
| `match` | 4202 verified pairs of 19900, largest connected component 160 images |
| `map` | **151/200 images registered, 838 points, 0.262px** — 276s, 40MB peak RSS |

The 49 unregistered views are ones where too few markers decoded to form a
verified pair; they are genuinely disconnected from the match graph, not
dropped by the mapper.

**Planar targets** (a printed board, a marker sheet, a grid on a screen) are
handled: both the essential matrix and linear PnP are mathematically
degenerate for coplanar points, so the pipeline detects that case and switches
to homography-based two-view initialization and homography-based PnP (Zhang's
construction). Without it a planar fiducial dataset does not reconstruct at
all. Note that *calibrating* from a single plane still needs the board tilted
substantially between shots — see [Known limitations](#known-limitations).

**Supply your intrinsics if you have them.** Fiducial-only reconstruction
recovers *poses* well but does not currently refine the shared focal length —
see [Known limitations](#known-limitations). For calibrated cameras, give it
the calibration and let it solve the geometry:

```toml
# sfm.toml
[[cameras]]
name   = "cam"
images = "*"
model  = "SIMPLE_RADIAL"
params = [1150.0, 640.0, 480.0, 0.0]
refine = false                        # trust the values you measured
```

For a rig of **fixed** cameras where you move the marker board between shots,
add `--merge-multicaps` to the `feature` step — see
[Quick start](#quick-start).

## Evaluating a reconstruction

`sfmtory eval` reads a COLMAP-format model and reports reprojection error
recomputed **from the geometry itself** (not read back from the model's stored
`error` column, so it also catches a model written out inconsistently, and is
comparable against another tool's output):

```bash
sfmtory eval                                    # this project's map output
sfmtory eval --ours path/to/sparse/0            # any COLMAP model
sfmtory eval --gt-focal 1523.15                 # focal error vs a known value
sfmtory eval --gt-focal data/sceaux_castle/K.txt   # ...or a K matrix file
sfmtory eval --baseline path/to/colmap/sparse/0    # compare against COLMAP
```

```text
Reprojection error (recomputed from geometry):
  ours       images   47  points   10599  obs    40996  mean 0.2831px  median 0.1438px  p95 0.8731px  max 572.070px
Focal lengths:
  camera 1    f =   1531.968 px   error vs reference  0.579%
```

Median and p95 alongside the mean are deliberate: a single mean hides whether
a model is uniformly decent or mostly excellent with a few broken points — the
`max` above is one such outlier that the mean alone would not reveal.

## GPU support

GPU acceleration is optional and auto-detected, added where it actually pays off:
classical SIFT/ORB extraction and brute-force matching are already fast enough on CPU
that GPU transfer overhead would net negative, and bundle adjustment's sparse solve
doesn't map cleanly onto GPU at all — so GPU only matters for *learned* detectors/
matchers. **`disk`** is implemented: a fully-convolutional learned keypoint detector, run
through [`ort`](https://github.com/pykeio/ort) (ONNX Runtime's Rust bindings).

```bash
sfmtory feature --detector disk --gpu
```

CUDA and TensorRT execution providers are tried first when `--gpu` is passed, falling
back to CPU automatically and silently if no compatible GPU/runtime is available —
`--gpu` never causes a hard failure on a CPU-only machine. Nothing is bundled: the
~4.4MB DISK model (Apache-2.0 licensed weights, `cvlab-epfl/disk`) is downloaded to a
per-user cache directory and checksum-verified on first use. On `sceaux_castle`, `disk`
finds dramatically more matches than SIFT (72004 vs. 17187 inlier matches, 55/55 vs.
49/55 verified pairs) and more 3D points (13919 vs. 8111), at the cost of somewhat worse
reprojection precision (0.691px vs. 0.397px — DISK's keypoints are read directly off a
heatmap grid with no sub-pixel refinement) — see `decisions.md`'s "GPU support" for the
full license research, a real memory-safety bug found and fixed during validation (large
photos could exhaust memory on CPU inference without a resolution cap), and the complete
numbers. SuperPoint and ALIKED are deliberately not offered: SuperPoint's original
weights are non-commercially licensed, and ALIKED's weight license isn't clearly
confirmed anywhere — neither meets this project's commercial-use-only dependency rule.
A GPU-capable LightGlue matcher (same license family, same `ort` infrastructure) is the
natural next step, not yet implemented.

## Known limitations

- **Point count on `sceaux_castle` is ~2% behind both** (7758 vs. COLMAP's 7927 and
  GLOMAP's 7851) — the one accuracy metric sfmtory doesn't lead on any dataset. It
  regressed slightly as a side effect of *better* calibration: outlier filtering
  against correct intrinsics legitimately rejects more observations. The fix is
  COLMAP's track completion/retriangulation after global bundles, which isn't
  implemented — see [`decisions.md`](decisions.md).
- **Map-stage wall-clock is ~8% behind COLMAP on `temple_ring`** (15.8s vs. 14.5s),
  while ~2x ahead of GLOMAP. The structural differences from COLMAP's mapper are
  closed; what's left is per-iteration constant factors — Ceres uses a sparse
  Cholesky where this uses a dense one on the reduced camera system. That's also
  what needs to change to scale past a few hundred images.
- **Self-calibration from a single planar target is unreliable.** Planar
  scenes now *reconstruct* (homography-based initialization and PnP), but with
  all structure on one plane the focal length trades off against the plane's
  pose, and many focals reproject equally well. On a 13-photo ArUco-on-a-screen
  dataset the recovered grid comes out neither flat nor square, and sweeping a
  fixed focal from 500 to 1536 px does not discriminate. Tilt the board
  substantially between shots (as Zhang's method requires), or supply known
  intrinsics with `refine = false`.
- **Fiducial-only datasets refine the shared focal length only weakly.** ArUco
  reconstruction recovers poses and structure well (151/200 views at 0.262px on
  a 200-view test scene), but every marker-corner track ends up with exactly
  two observations, and a two-view track carries no redundancy to constrain a
  shared intrinsic. The focal therefore stays at its initial estimate. Pass
  known intrinsics with `refine = false` (see
  [Calibrating from ArUco markers](#calibrating-from-aruco-markers)) until
  track completion covers the fiducial path — this is the same track-merging
  gap noted below, where it happens to bite hardest.
- **`temple_sparse_ring`'s self-calibration is delicate.** It now beats both real
  systems there (0.82% focal error vs. 3.82%/1.90%), but it's the dataset that flips
  between good and bad optimization basins on small changes to bundle-adjustment
  scheduling. Treat it as the canary for any change in that area — there's a measured
  sensitivity table in [`decisions.md`](decisions.md).
- Camera intrinsics (focal length + distortion) *are* refined jointly with poses/points,
  with real safeguards (principal point held fixed, a minimum-images-per-camera gate, an
  implausible-distortion check, and outlier-observation filtering) added after real test
  data exposed exactly the failure modes those safeguards close. See
  [`decisions.md`](decisions.md) for the full story if you're touching `sfm-ba` or
  `sfm-reconstruction`'s bundle-adjustment code - several of its design choices look
  arbitrary until you know what they prevent.
- `sfmtory map --pipeline global`'s own accuracy still trails `--pipeline incremental`'s on
  real data (no cross-component stitching or retriangulation yet) — `incremental` is the
  one benchmarked above; `global` is implemented and much faster, but not yet the
  recommended default for accuracy-sensitive use.
- `sfmtory refine` and `sfmtory eval` are typed CLI stubs; the underlying `sfm-ba` and
  reconstruction-loading logic they'd use already exist and work.
- Spatial and ArUco-covisibility pairing aren't implemented (vocabulary-tree is).
- GPU support covers the `disk` detector only (see [GPU support](#gpu-support)); `sfm
  match --gpu` is still a no-op pending a LightGlue matcher implementation.
- Mapping-stage timing, while much improved (see [GPU support](#gpu-support) intro and
  `decisions.md`'s "Mapping-stage timing investigation"), is still ~4.5x behind COLMAP's
  own incremental mapper on `temple_ring` — the remaining cost is understood (a single
  ill-conditioned bundle-adjustment solve, not low-hanging fruit) but not yet closed.

Full detail, with reasoning for each: [PLAN.md](PLAN.md).

## Roadmap

- Closing the global pipeline's accuracy gap vs. incremental (cross-component
  stitching, retriangulation).
- A GPU-capable LightGlue matcher (`sfmtory match --detector disk --matcher lightglue --gpu`),
  reusing the same `ort`/model-cache infrastructure `disk` already added.
- AKAZE.
- Spatial / ArUco-covisibility image pairing (vocabulary-tree retrieval is done).
- A sparse/block-sparse reduced-system solve in `sfm-ba` — closes the last of the
  map-stage timing gap vs. COLMAP and is what's needed to scale past a few hundred
  images (see [Known limitations](#known-limitations)).
- COLMAP-style track completion / retriangulation after global bundles, to close the
  `sceaux_castle` point-count gap.

See [PLAN.md](PLAN.md) for the full, up-to-date checklist.

## Contributing

This is an active work in progress developed against [PLAN.md](PLAN.md)'s checklist.
Before sending a change, check whether the relevant crate's module docs already explain
why something is the way it is — several apparent gaps are documented, deliberate
tradeoffs rather than oversights.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
