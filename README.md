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
> end-to-end today and is unit-tested at every layer (37 tests across 8 crates).
> **Benchmarked against real COLMAP on real photos, across three datasets** (11, 16, and
> 47 images). sfmtory **matches or beats COLMAP's registration count on all three**
> (11/11, 16/16 beating COLMAP's 13/16, and 47/47), and on the largest/best-conditioned
> dataset (`temple_ring`, 47 images) **beats COLMAP outright on every metric** -
> registration, reprojection error (0.30px vs. 0.32px), and focal length accuracy (1.2%
> vs. 2.0%). On the two smaller datasets, focal-length accuracy is still somewhat behind
> COLMAP's. See [Status vs. COLMAP](#status-vs-colmap) below for the real numbers and
> caveats before trusting this for anything production-critical. [PLAN.md](PLAN.md)
> tracks what's implemented and what's stubbed; [decisions.md](decisions.md) has the
> design rationale.

## Contents

- [Why](#why)
- [Features](#features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Status vs. COLMAP](#status-vs-colmap)
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
  ORB, and a custom square fiducial-marker ("ArUco-style") detector for rig calibration
  with high-confidence correspondences. AKAZE and learned detectors (SuperPoint/DISK) are
  planned but not implemented yet.
- **Matching**: mutual-nearest-neighbor + Lowe's ratio test, RANSAC-verified two-view
  geometry (normalized 8-point + adaptive RANSAC + local optimization), with weak/false
  pairs rejected outright rather than fed downstream.
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
cargo build --release
# binary at target/release/sfm
```

Run the full pipeline on a folder of images in one command:

```bash
sfm project new my_project --images /path/to/photos
sfm run --project my_project --pipeline incremental
```

`my_project/export/` now holds a COLMAP text model. See the
**[User Guide](docs/USER_GUIDE.md)** for a step-by-step tutorial (running each stage
individually, reading the output, exporting to NeRF format, and troubleshooting).

No photos handy? [`data/README.md`](data/README.md) has download commands for two small
public test sets, and the User Guide's
[Testing with the sample datasets](docs/USER_GUIDE.md#testing-with-the-sample-datasets)
section has exact commands and real (not aspirational) example output for both.

## Architecture

A Cargo workspace, one crate per pipeline concern:

| Crate | Responsibility |
|---|---|
| `sfm-core` | Camera models, poses, the sparse reconstruction data model |
| `sfm-io` | COLMAP text and NeRF `transforms.json` readers/writers |
| `sfm-features` | SIFT / ORB / ArUco-style detectors |
| `sfm-geometry` | Two-view geometry, PnP, triangulation, RANSAC |
| `sfm-match` | Descriptor matching + geometric verification |
| `sfm-ba` | Bundle adjustment (Levenberg-Marquardt, Schur complement) |
| `sfm-reconstruction` | The incremental SfM engine tying the above together |
| `sfm-cli` | The `sfm` binary |

Each crate's module docs explain its own deliberate simplifications (e.g. numerical vs.
analytic Jacobians in `sfm-ba`, linear vs. minimal solvers in `sfm-geometry`) — read those
before assuming a shortcut is a bug.

## Status vs. COLMAP

The original goal is to match or beat COLMAP's calibration accuracy *and* registration
completeness. Real COLMAP (via `pycolmap`) has been run head-to-head against sfmtory on
the same real photos for all three test sets in [`data/`](data/README.md):

| | sfmtory | COLMAP |
|---|---|---|
| `sceaux_castle` (11 images) registered | **11/11** | 11/11 |
| `sceaux_castle` mean reprojection error | **0.42px** | 0.62px |
| `sceaux_castle` focal length error vs. known truth | 3.1% | **2.3%** |
| `temple_sparse_ring` (16 images) registered | **16/16** | 13/16 |
| `temple_sparse_ring` mean reprojection error | 0.38px | **0.22px** |
| `temple_sparse_ring` focal length error vs. known truth | 3.7% | **0.02%** |
| `temple_ring` (47 images) registered | 47/47 | 47/47 |
| `temple_ring` mean reprojection error | **0.30px** | 0.32px |
| `temple_ring` focal length error vs. known truth | **1.2%** | 2.0% |

**sfmtory matches or beats COLMAP's registration count on all three datasets**, and on
the largest/best-conditioned one (`temple_ring`, 47 images) **beats COLMAP outright on
every metric** — see [`decisions.md`](decisions.md) for the root causes found and fixed
to close what used to be a large registration-count gap (non-deterministic RANSAC
sampling, a degenerate-PnP-sample gap, a missing nonlinear pose refinement, disconnected
seed selection, a chain-graph bootstrap fallback, and SIFT upsampling for small images).
The honest remaining gap is **focal length accuracy on the two smaller datasets**: still
somewhat behind COLMAP's on `sceaux_castle` (3.1% vs. 2.3%) and `temple_sparse_ring`
(3.7% vs. 0.02%) — see [Known limitations](#known-limitations) below. There's no
`BENCHMARKS.md` write-up yet and `sfm eval`'s automated comparison logic is still a stub
(this comparison was run by hand against real COLMAP output), so treat this as real,
reproducible data points rather than a comprehensive benchmark suite.

## Known limitations

- **Focal length error is behind COLMAP's on the two smaller datasets** (3.1% vs. 2.3%
  on `sceaux_castle`, 3.7% vs. 0.02% on `temple_sparse_ring`), despite registration
  count and reprojection error now matching or beating COLMAP on all three, and despite
  beating COLMAP on focal length too on the largest dataset (`temple_ring`, 47 images:
  1.2% vs. 2.0%). An outlier-filtering pass in the final bundle adjustment closed most
  of `sceaux_castle`'s gap (was 5.4%), and switching from numerical to exact analytic
  Jacobians for every camera model closed most of `temple_sparse_ring`'s (was 6.3%) -
  see [`decisions.md`](decisions.md)'s "Known open gaps" for what's been tried and ruled
  out on the remainder.
- `sfm map --pipeline global` (the eventually-intended default, GLOMAP-style) isn't
  implemented — pass `--pipeline incremental` explicitly.
- Camera intrinsics (focal length + distortion) *are* refined jointly with poses/points
  now, with real safeguards (principal point held fixed, a minimum-images-per-camera
  gate, an implausible-distortion check, and outlier-observation filtering) added after
  real test data exposed exactly the failure modes those safeguards close. See
  [`decisions.md`](decisions.md) for the full story if you're touching `sfm-ba` or
  `sfm-reconstruction`'s bundle-adjustment code - several of its design choices look
  arbitrary until you know what they prevent.
- `sfm refine` and `sfm eval` are typed CLI stubs; the underlying `sfm-ba` and
  reconstruction-loading logic they'd use already exist and work.
- Image pairing is exhaustive or sequential-window only; vocabulary-tree and spatial
  pairing (needed to scale past a few hundred images) aren't implemented.
- No GPU path yet. `--gpu` flags exist on `extract`/`match` for forward compatibility but
  currently have no effect (there's nothing GPU-accelerated to switch to — all detectors
  and the matcher are classical CPU algorithms right now).

Full detail, with reasoning for each: [PLAN.md](PLAN.md).

## Roadmap

- **Optional GUI** — a viewer/front-end on top of the existing CLI pipeline, for
  inspecting reconstructions (camera poses, sparse point cloud) and driving
  extract/match/map/export without hand-typing commands. The CLI and on-disk project
  format remain the primary, scriptable interface either way.
- `sfm map --pipeline global` (GLOMAP-style global SfM).
- AKAZE and learned feature detectors (SuperPoint/DISK).
- Vocabulary-tree / spatial image pairing for datasets beyond a few hundred images.
- Closing the remaining focal-length accuracy gap on smaller datasets (see
  [Known limitations](#known-limitations)).

See [PLAN.md](PLAN.md) for the full, up-to-date checklist.

## Contributing

This is an active work in progress developed against [PLAN.md](PLAN.md)'s checklist.
Before sending a change, check whether the relevant crate's module docs already explain
why something is the way it is — several apparent gaps are documented, deliberate
tradeoffs rather than oversights.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
