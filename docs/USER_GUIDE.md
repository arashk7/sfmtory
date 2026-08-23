# sfmtory User Guide

A hands-on walkthrough of the `sfm` CLI: installing it, running the full pipeline on a
real photo set, understanding what each stage produces, and reading the results. For
what's implemented vs. still a stub, see the [README](../README.md#known-limitations)
and [PLAN.md](../PLAN.md).

## Table of contents

1. [Installation](#installation)
2. [Core concepts](#core-concepts)
3. [Tutorial: from photos to a 3D reconstruction](#tutorial-from-photos-to-a-3d-reconstruction)
4. [Testing with the sample datasets](#testing-with-the-sample-datasets)
5. [Command reference](#command-reference)
6. [Choosing a detector, pairing strategy, and pipeline](#choosing-a-detector-pairing-strategy-and-pipeline)
7. [Calibrating with ArUco-style markers](#calibrating-with-aruco-style-markers)
8. [Reading the output](#reading-the-output)
9. [Troubleshooting](#troubleshooting)

## Installation

You need a recent stable Rust toolchain. If you don't have one:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then build sfmtory:

```bash
git clone <this-repo>
cd sfmtory
cargo build --release
```

The binary is at `target/release/sfm`. The examples below assume it's on your `PATH`
(`export PATH="$PWD/target/release:$PATH"`) or you can prefix every command with
`./target/release/`.

Verify it works:

```bash
sfm --help
```

## Core concepts

**A project** is a directory sfmtory manages for you, created by `sfm project new`. It
contains:

```
my_project/
  sfm.toml            # points at your images directory
  database.sqlite      # detected features, verified matches — inspectable with any
                        # off-the-shelf SQLite browser
  sparse/0/             # the reconstruction: cameras.txt, images.txt, points3D.txt
  export/                # where `sfm export` writes final deliverables
  logs/                  # one JSON file per pipeline run, e.g. logs/extract_1699999999.json
```

**The pipeline** is four stages, each its own subcommand, each reading the previous
stage's output from `database.sqlite` or `sparse/0/`:

```
sfm extract   →   sfm match   →   sfm map   →   sfm export
(detect         (find & verify    (recover        (write COLMAP text
 keypoints)      correspondences)  camera poses     or NeRF transforms.json)
                                    + 3D points)
```

`sfm run` chains all four for the common case. Running stages individually is useful when
you want to try a different detector without re-matching, inspect intermediate state, or
resume after a crash.

## Tutorial: from photos to a 3D reconstruction

### 1. Get a photo set

Take 20-100 overlapping photos walking around a static subject — a building, a room, an
object on a table. Practical tips that matter more for this early-stage pipeline than they
would for COLMAP:

- **Aim for 60-80% overlap** between consecutive photos. `sfm match --pairing sequential`
  (the default) only checks nearby photos in filename order, so shoot them in a
  roughly continuous path and let your file names sort in capture order.
- **Avoid a purely flat/planar subject** (a poster, a flat facade shot head-on) if you can
  — a scene with real depth variation gives much better-conditioned two-view geometry and
  camera registration. A perfectly flat scene is a textbook degenerate case for the linear
  PnP solver sfmtory currently uses (see [PLAN.md](../PLAN.md) §1, `sfm-geometry`).
- **Avoid pure in-place rotation** (spinning the camera without translating) — sfmtory (like
  any SfM system) recovers structure from parallax, which needs the camera to actually move
  through space, not just turn.
- JPEG/PNG/TIFF/BMP are all supported.

### 2. Create a project

```bash
sfm project new my_project --images ./photos
```

```
Created project at my_project (images: /abs/path/to/photos)
Next: sfm extract --project my_project
```

### 3. Run the pipeline

The fastest path is one command. **Note:** `--pipeline incremental` must be passed
explicitly right now — `global` is the eventual default per the roadmap but isn't
implemented yet, and will error out if you don't override it.

```bash
sfm run --project my_project --pipeline incremental
```

This runs `extract` (SIFT by default) → `match` (sequential pairing, mutual-NN matching)
→ `map` (incremental reconstruction) → `export` (COLMAP text, to `my_project/export/`).
Expect output shaped like this (real numbers from the sample datasets, not this
made-up image/feature count, are in the [next section](#testing-with-the-sample-datasets)):

```
`sfm run` chains extract -> match -> map -> export; running each stage now.
Extracted 8214 features across 24 images (Sift). Logged to my_project/logs/extract_....json.
Verified 20/46 pairs, 3100 inlier matches total. Logged to my_project/logs/match_....json.
Registered 11/24 images, 1900 points3d, mean reprojection error 0.55px. Wrote my_project/sparse/0. Logged to my_project/logs/map_....json.
Exported 11 images / 1900 points to my_project/export
```

**Don't be surprised if well under half your images register.** This is a real,
current limitation (not a misconfiguration) in how new images get registered during
reconstruction — see the [next section](#testing-with-the-sample-datasets) and
[PLAN.md](../PLAN.md) for exactly why and what's being done about it. `sfm map`'s
summary line always tells you the true count (`Registered 11/24 images` above). Check
`mean reprojection error` for what *did* register: under ~1-2 pixels is a good sign;
several pixels or more suggests noisy matches (see [Troubleshooting](#troubleshooting)).

### 4. Or run it stage by stage

Useful when you want to inspect intermediate output or try different settings without
redoing earlier work:

```bash
sfm extract --project my_project --detector sift --max-features 4000
sfm match --project my_project --pairing exhaustive
sfm map --project my_project --pipeline incremental
sfm export --project my_project --format colmap-text
```

Every stage writes a timestamped JSON report to `my_project/logs/` regardless of how you
invoke it, so you always have a record of what ran and with what result.

### 5. Export to NeRF format instead

If you're feeding the reconstruction into a NeRF/Gaussian-splatting pipeline
(nerfstudio, instant-ngp) rather than a COLMAP-format consumer:

```bash
sfm export --project my_project --format nerf-transforms
cat my_project/export/transforms.json
```

## Testing with the sample datasets

`data/` (git-ignored — see [`data/README.md`](../data/README.md)) holds two small,
real, publicly-hosted test sets you can run the pipeline against immediately, without
taking your own photos. Fetch them with the download commands in `data/README.md`, then:

### Château de Sceaux (real outdoor photos)

```bash
sfm project new /tmp/castle --images data/sceaux_castle/images
sfm extract --project /tmp/castle --detector sift
sfm match --project /tmp/castle --pairing exhaustive
sfm map --project /tmp/castle --pipeline incremental
sfm export --project /tmp/castle --format colmap-text
```

(`--pairing exhaustive` rather than the default `sequential` here: with only 11 images
the O(n²) cost is trivial, and exhaustive checking finds every viable pair instead of
only nearby-numbered ones.)

As of this writing, expect:

```
Extracted 56378 features across 11 images (Sift).
Verified 41/55 pairs, 7654 inlier matches total.
Registered 11/11 images, 2869 points3d, mean reprojection error 0.427px.
Exported 11 images / 2869 points to /tmp/castle/export
```

Check `/tmp/castle/export/cameras.txt`'s focal length against the dataset's known-true
value in `data/sceaux_castle/K.txt` (`2905.88`): sfmtory recovers something in the same
ballpark (currently ~3.1% off, vs. real COLMAP's own 2.3% on the same photos - close but
not yet matching; see [`decisions.md`](../decisions.md)'s "Known open gaps"). `sfm-ba`'s
bundle adjustment jointly refines focal length and distortion alongside poses/points,
with the principal point (`cx`/`cy`) deliberately held fixed, and filters out
high-reprojection-error observations before its final refinement pass specifically so a
handful of noisier points can't bias the shared focal length - see `decisions.md` for
the full rationale.

**All 11 images register, matching real COLMAP's count on this same set** - this took
several real-data-driven fixes to the incremental registration pipeline (non-
deterministic RANSAC sampling, linear PnP-DLT accepting degenerate near-coplanar
samples, a missing nonlinear reprojection-error pose refinement, connected-component-
aware seed selection, and a bootstrap fallback for images whose match-graph
connectivity is too thin for ordinary PnP - see `decisions.md` for the full diagnosis of
each). Mean reprojection error (0.42px) is meaningfully better than COLMAP's 0.62px.

### Middlebury temple (small object, unusual sparse view sampling)

```bash
sfm project new /tmp/temple --images data/temple_sparse_ring/images
sfm extract --project /tmp/temple --detector sift
sfm match --project /tmp/temple --pairing exhaustive
sfm map --project /tmp/temple --pipeline incremental
```

Expect:

```
Extracted 12736 features across 16 images (Sift).
Verified 21/120 pairs, 2194 inlier matches total.
Registered 16/16 images, 921 points3d, mean reprojection error 0.378px.
```

**All 16 images register, beating real COLMAP's 13/16 on this same set.** This dataset's
small, low-texture 640x480 photos originally extracted very few SIFT features (~6000
across all 16, versus ~12700 now) - enabling SIFT's Lowe-original 2x pre-upsampling for
small images fixed that, which in turn gave the match graph enough density and triangle
redundancy (rather than a near-linear chain) for ordinary PnP and the bootstrap fallback
to reach every image. Focal length is still further behind COLMAP here than on
`sceaux_castle` (~3.7% off vs. COLMAP's ~0.02%) - see `decisions.md`'s "Known open gaps".

For a rigorous accuracy check rather than eyeballing reprojection error, compare
`/tmp/temple/sparse/0/images.txt`'s recovered poses against `data/temple_sparse_ring/
templeSR_par.txt`'s ground truth for whichever images registered (matching by
filename) — keeping in mind sfmtory's reconstruction is in an arbitrary scale/frame
relative to the seed image, so a direct comparison needs a similarity-transform
alignment first (this is exactly what `sfm eval`'s Umeyama-alignment logic will
automate once it's implemented — see [PLAN.md](../PLAN.md) §7).

### Middlebury temple ring (same object, 47 images, a full walk-around)

```bash
sfm project new /tmp/ring --images data/temple_ring/images
sfm extract --project /tmp/ring --detector sift
sfm match --project /tmp/ring --pairing exhaustive
sfm map --project /tmp/ring --pipeline incremental
```

Expect:

```
Extracted 37786 features across 47 images (Sift).
Verified 246/1081 pairs, 34768 inlier matches total.
Registered 47/47 images, 6647 points3d, mean reprojection error 0.298px.
```

**All 47 images register, tying real COLMAP's 47/47 - and sfmtory beats COLMAP outright
on both other metrics here**: 0.30px mean reprojection error vs. COLMAP's 0.32px, and
1.2% focal length error vs. COLMAP's 2.0% (recovered focal 1504.8 vs. the dataset's true
~1523.15). This is the same physical object/rig/intrinsics as `temple_sparse_ring`
above, just a real ring capture instead of that dataset's sparse two-latitude-band
sampling - more images gives self-calibration a genuinely better-conditioned problem to
solve, and both solvers' accuracy improves accordingly. Takes noticeably longer than the
other two datasets (a couple of minutes on modest hardware) since seed selection tries
several candidate pairs and fully grows a trial reconstruction from each before picking
the best (see `decisions.md`'s "Seed & registration-graph structure").



All commands take `--project <dir>` pointing at a project created with `sfm project new`,
except `sfm project new` itself and `sfm eval`.

### `sfm project new <dir> --images <path>`

Scaffolds a new project. `<dir>` is created; `--images` must already exist and contain
your photos.

### `sfm extract --project <dir> [options]`

Detects keypoints/descriptors for every image and stores them in `database.sqlite`.
Images are automatically grouped into shared cameras by pixel resolution (so 40 photos
all 4032×3024 share one set of intrinsics, as they should if shot on the same device).

| Flag | Default | Notes |
|---|---|---|
| `--detector <sift\|orb\|aruco\|akaze\|superpoint\|disk>` | `sift` | Only `sift`, `orb`, `aruco` are implemented; the other three error out clearly. |
| `--max-features <N>` | unlimited | Keeps the N highest-response keypoints per image. |
| `--aruco-dict <name>` | — | Accepted but currently unused; sfmtory's ArUco detector uses its own generated dictionary regardless (see [Calibrating with ArUco-style markers](#calibrating-with-aruco-style-markers)). |
| `--gpu` | off | Accepted for forward compatibility; has no effect today (no GPU-accelerated detector exists yet). |

Re-running `sfm extract` on the same project reuses existing camera/image rows rather
than duplicating them, so it's safe to re-extract with different settings.

### `sfm match --project <dir> [options]`

Pairs images, matches their descriptors, and geometrically verifies each pair with
RANSAC, storing only pairs that pass a minimum inlier count/ratio.

| Flag | Default | Notes |
|---|---|---|
| `--pairing <exhaustive\|sequential\|spatial\|vocab-tree\|aruco>` | `sequential` | Only `exhaustive` and `sequential` are implemented. `exhaustive` checks every pair (O(n²) — fine under a few hundred images); `sequential` only checks images within `--window` positions of each other in the (filename-)sorted image list. |
| `--matcher <mnn-ratio\|lightglue>` | `mnn-ratio` | Only `mnn-ratio` (mutual-nearest-neighbor + Lowe's ratio test) is implemented. |
| `--window <N>` | `10` | Only used with `--pairing sequential`. |
| `--gpu` | off | Same as `extract --gpu`: accepted, currently a no-op. |

### `sfm map --project <dir> [options]`

Runs sparse reconstruction from the verified pairs and writes `<project>/sparse/0/`.

| Flag | Default | Notes |
|---|---|---|
| `--pipeline <global\|incremental>` | `global` | **Only `incremental` is implemented** — `global` (a faster, GLOMAP-style pipeline) is on the roadmap but errors out today. Pass `--pipeline incremental` explicitly. |

### `sfm export --project <dir> --format <format> [--out <path>]`

Reads `<project>/sparse/0/` and writes it in another format.

| Flag | Default | Notes |
|---|---|---|
| `--format <colmap-text\|nerf-transforms>` | required | — |
| `--out <path>` | `<project>/export/` | For `nerf-transforms`, a directory path gets `transforms.json` appended; a path with a file extension is used as-is. |

### `sfm run --project <dir> [options]`

Chains `extract` → `match` → `map` → `export` (COLMAP text, to the default export
directory). Accepts `--detector`, `--pairing`, `--matcher`, `--pipeline` — same meaning
and defaults as the individual commands above (so `--pipeline incremental` still needs
to be passed explicitly).

### `sfm refine` / `sfm eval`

Both are currently typed CLI stubs — they parse their arguments and print a
"not implemented yet" note rather than doing the described work. `sfm eval` in its
current form will load and print basic stats (image/point counts, mean reprojection
error) for `--ours` and, if given, `--baseline`, but does not yet perform the intended
pose-accuracy comparison.

## Choosing a detector, pairing strategy, and pipeline

**Detector** (`sfm extract --detector`):
- `sift` (default) — best general-purpose choice for photos of textured real-world
  scenes; scale- and rotation-invariant.
- `orb` — faster, binary descriptors; a reasonable choice if `sift` is too slow on a very
  large image set, at some cost to matching robustness under scale/illumination change.
- `aruco` — use this *instead of* `sift`/`orb` only when your scene contains printed
  fiducial markers and you specifically want marker-corner correspondences (see below);
  it will not find ordinary image features.

**Pairing** (`sfm match --pairing`):
- `sequential` (default) — the right choice for a photo *walk* (video frames, or stills
  shot while moving continuously around a subject) where nearby-in-sequence images are
  the ones that actually overlap.
- `exhaustive` — the right choice for an unordered photo *set* (e.g. downloaded from
  multiple sources, or shuffled) where you can't assume filename order reflects spatial
  proximity. Only scales to roughly a few hundred images before match time becomes
  painful (it's O(n²) in image count).

**Pipeline** (`sfm map --pipeline`): only `incremental` exists right now, so there's no
real choice to make yet — see [Known limitations](../README.md#known-limitations).

## Calibrating with ArUco-style markers

sfmtory includes a native square fiducial-marker detector for rigs where you want very
high-confidence correspondences (e.g. calibrating a multi-camera rig against printed
markers on a wall or calibration board) rather than relying on scene texture.

**Important:** this is *not* byte-compatible with OpenCV's standard ArUco dictionaries
(`DICT_4X4_50` etc). It uses its own dictionary of 50 generated 4×4-bit codes with a
minimum Hamming distance for reliable identification. You cannot print an OpenCV ArUco
sheet and expect it to be recognized — markers must be generated from sfmtory's own
dictionary (`sfm_features::aruco::dictionary()`; a `sfm features print-markers` CLI
command to make this convenient without writing Rust is on the roadmap but doesn't exist
yet).

To use it:

```bash
sfm extract --project my_project --detector aruco
sfm match --project my_project --pairing exhaustive
```

Each detected marker contributes its 4 corners as keypoints, matched across images by
exact `(marker_id, corner_index)` identity rather than descriptor distance — so as long
as the same physical marker is visible in two images, sfmtory will find the
correspondence with no ambiguity.

## Reading the output

### COLMAP text format (`sparse/0/`, or `export/` with `--format colmap-text`)

Three files, COLMAP's own format:

- **`cameras.txt`**: one line per physical camera — `CAMERA_ID MODEL WIDTH HEIGHT
  PARAMS...`. `MODEL` is one of `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`,
  `OPENCV`, `OPENCV_FISHEYE`.
- **`images.txt`**: two lines per registered image — a header (`IMAGE_ID QW QX QY QZ TX
  TY TZ CAMERA_ID NAME`, quaternion + translation is the **world-to-camera** transform,
  Hamilton convention) followed by a line of `X Y POINT3D_ID` triples (one per detected
  keypoint; `POINT3D_ID` is `-1` if that keypoint wasn't triangulated).
- **`points3D.txt`**: one line per 3D point — `POINT3D_ID X Y Z R G B ERROR TRACK...`,
  where `TRACK` lists every `(IMAGE_ID, POINT2D_IDX)` observation of that point. Point
  colors are currently a fixed gray placeholder (`180 180 180`) — color isn't sampled
  from the source images yet.

Because this is a direct implementation of COLMAP's own documented format, any tool that
reads a COLMAP sparse model (Meshlab, Blender's COLMAP importer, nerfstudio's COLMAP
dataparser, COLMAP itself) should accept sfmtory's output as-is.

### NeRF `transforms.json` (`export/transforms.json` with `--format nerf-transforms`)

Follows the nerfstudio/instant-ngp convention: `fl_x`/`fl_y`/`cx`/`cy`/`w`/`h` (and
`k1`/`k2`/`p1`/`p2` distortion) either once at the top level (if every image shares one
camera) or per-frame, plus a `frames` array of `{file_path, transform_matrix}`.
`transform_matrix` is a **camera-to-world** 4×4 matrix in OpenGL/Blender axis convention
(+X right, +Y up, +Z backward) — sfmtory converts from its own internal OpenCV-convention
world-to-camera poses for you, including the axis flip, which is the most common source
of bugs in hand-rolled COLMAP↔NeRF converters.

## Troubleshooting

**`no images found in <dir>`** — `sfm extract`'s `--images` (set at `project new` time)
found no files with a recognized extension (`.jpg .jpeg .png .tif .tiff .bmp`). Check the
path in `<project>/sfm.toml`.

**`need at least 2 images with extracted features to match; run 'sfm extract' first`** /
**`... to map; run 'sfm extract' first`** — you ran `match` or `map` before `extract`, or
`extract` found fewer than 2 images.

**`no verified image pairs found; run 'sfm match' first'`** — either you haven't run
`sfm match` yet, or every pair failed geometric verification. If it's the latter: check
your `--pairing` choice matches your capture style (see
[above](#choosing-a-detector-pairing-strategy-and-pipeline)) — a `sequential` pairing
with too-small `--window` on a photo set that isn't actually filename-ordered by capture
sequence will find no real overlapping pairs.

**`reconstruction failed to register any images; check 'sfm match' results (too few/weak
verified pairs?)`** — `sfm map` found verified pairs but couldn't build a usable seed
reconstruction from them (too few inlier matches even in the best pair). Try
`--pairing exhaustive` to make sure you're not missing valid pairs, or check that your
photos have genuine 3D parallax (see the [tutorial](#1-get-a-photo-set) tips on avoiding
degenerate/flat scenes).

**Only some images register (`Registered 18/24 images`, or well under half)** — an
image is left unregistered either because it genuinely lacks 2D-3D correspondences
against already-registered images (default minimum: 12 — check that image against its
neighbors in the capture sequence for missing overlap), *or* because of a real current
limitation: the linear PnP solver used to register each new image is weaker than a
proper minimal solver specifically when the already-triangulated points it has to work
with are close to coplanar, which a short-baseline photo walk produces easily
regardless of the scene's actual 3D structure. See [Testing with the sample
datasets](#testing-with-the-sample-datasets) for concrete before/after numbers and
[PLAN.md](../PLAN.md) for the fix priority. If most of your set has good overlap and
registration still stalls early, this is very likely why — not a sign your photos or
command are wrong.

**High mean reprojection error (several pixels or more)** — points either from a poorly-
conditioned scene (see the degenerate-scene tips above) or from an initial camera
intrinsics guess that's far off. `sfm extract` seeds focal length from image dimensions
alone (no EXIF reading yet) as a starting guess for bundle adjustment to refine — a very
unusual lens (unusually wide or long focal length) may need more registered images before
BA converges to accurate intrinsics.

**`detector <X> is not implemented yet`**, **`matcher <X> is not implemented yet`**,
**`pairing <X> is not implemented yet`**, **`pipeline <X> is not implemented yet`** — you
asked for an option that's on the roadmap but not built. The error message always lists
what *is* available; see [Known limitations](../README.md#known-limitations) for the full
list and status.
