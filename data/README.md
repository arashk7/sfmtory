# Test datasets

Three small, well-known, publicly-hosted multi-view image sets for exercising the
sfmtory pipeline end-to-end. None are redistributed with this repository — `data/` is
git-ignored (see `.gitignore`); re-download with the commands below whenever you need
them. See [`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md#testing-with-the-sample-datasets)
for exact `sfm` commands to run against each.

## `sceaux_castle/` — Château de Sceaux

11 photos (~1-1.8 MB each, ~13 MB total) walking around a stone castle facade. A classic
small SfM demo set distributed by the [openMVG](https://github.com/openMVG/openMVG)
project specifically for its own SfM tutorials.

- Source: <https://github.com/openMVG/ImageDataset_SceauxCastle>
- Photographer: © 2012 Pierre Moulon, publicly distributed for SfM tutorial/testing use.
- `K.txt` gives the known camera intrinsics (`fx=fy=2905.88, cx=1416, cy=1064` for the
  full-resolution ~2832×2128 images) as an independent sanity check on whatever focal
  length sfmtory's bundle adjustment converges to.
- Re-download:
  ```bash
  mkdir -p data/sceaux_castle/images && cd data/sceaux_castle/images
  for i in 7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110; do
    curl -sSO "https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/master/images/100_${i}.JPG"
  done
  curl -sS -o ../K.txt "https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/master/images/K.txt"
  ```

## `temple_sparse_ring/` — Middlebury "templeSparseRing"

16 photos (640×480 PNG, ~4 MB total) of a plaster replica of the Temple of the Dioskouroi,
captured with the Stanford spherical light-field gantry. Part of the
[Middlebury Multi-View Stereo](https://vision.middlebury.edu/mview/data/) benchmark suite
(Seitz, Diebel, Scharstein, Curless, Szeliski) — provided by Middlebury for research/
educational use; treat it accordingly (don't redistribute commercially, which is exactly
why it stays out of git history here).

**Despite the name, this is not a smooth walk-around ring** — check
`templeSR_ang.txt` and you'll find the 16 views sit in two narrow latitude bands
(elevation ≈ ±82°, i.e. looking almost straight down/up at the object) with scattered
longitudes, not an even sweep. It's a deliberately *sparse and unusual* view sampling
designed to stress-test MVS reconstruction, not a beginner-friendly capture pattern. Two
of the 16 images (`templeSR0001.png`/`templeSR0011.png`) are near-duplicate viewpoints
(both sit at longitude ±180°) — don't be surprised to see them match almost perfectly
while most other pairs match poorly or not at all. See [Testing with the sample
datasets](../docs/USER_GUIDE.md#testing-with-the-sample-datasets) for what sfmtory
actually recovers from this.

- Source: <https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip>
- `templeSR_par.txt` gives **ground-truth intrinsics and extrinsics for every image**
  (`imgname k11 k12 k13 k21 k22 k23 k31 k32 k33  r11..r33  t1 t2 t3`, projection matrix
  `K*[R|t]`) — genuinely useful for checking recovered poses against a known-correct
  answer, not just internal reprojection-error self-consistency.
- Re-download:
  ```bash
  mkdir -p data/temple_sparse_ring && cd data/temple_sparse_ring
  curl -sS -o templeSparseRing.zip "https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip"
  unzip -q templeSparseRing.zip
  mkdir -p images
  mv templeSparseRing/*.png images/
  mv templeSparseRing/README.txt ./README_dataset.txt
  mv templeSparseRing/templeSR_ang.txt templeSparseRing/templeSR_par.txt ./
  rmdir templeSparseRing
  rm templeSparseRing.zip
  ```

## `temple_ring/` — Middlebury "templeRing"

47 photos (640×480 PNG, ~11 MB total) — the same physical object, rig, and intrinsics as
`temple_sparse_ring` above, but a real walk-around ring capture (not that dataset's sparse
two-latitude-band sampling), added specifically to test at a larger image count than the
other two datasets (11 and 16 images) support.

- Source: <https://vision.middlebury.edu/mview/data/data/templeRing.zip>
- `templeR_par.txt` gives the same per-image ground-truth format as `temple_sparse_ring`'s
  `templeSR_par.txt`.
- Re-download:
  ```bash
  mkdir -p data/temple_ring && cd data/temple_ring
  curl -sS -o templeRing.zip "https://vision.middlebury.edu/mview/data/data/templeRing.zip"
  unzip -q templeRing.zip
  mkdir -p images
  mv templeRing/*.png images/
  mv templeRing/README.txt ./README_dataset.txt
  mv templeRing/templeR_ang.txt templeRing/templeR_par.txt ./
  rmdir templeRing
  rm templeRing.zip
  ```

## Why three, and why these three

They're deliberately different capture styles and scales, which exercises different
parts of the pipeline — and, honestly, exposed real bugs in sfmtory's reconstruction
stage the first time they were tried (see [`decisions.md`](../decisions.md) for what
those bugs were and how they were fixed):

| | sceaux_castle | temple_sparse_ring | temple_ring |
|---|---|---|---|
| Scene | Outdoor building facade | Small object, sparse/unusual view sampling | Small object, full ring walk-around |
| Images | 11, full-res photos | 16, 640×480 | 47, 640×480 |
| Capture path | Roughly linear walk | Two narrow latitude bands, scattered longitudes | Smooth ring |
| Ground truth | Intrinsics only (`K.txt`) | Full intrinsics + extrinsics per image | Full intrinsics + extrinsics per image |
| Good for testing | `--pairing exhaustive`, realistic photo noise/EXIF-less intrinsics guess | `--pairing exhaustive` (small enough to be cheap), quantitative pose-accuracy comparison | Larger-scale exhaustive matching/mapping, well-conditioned self-calibration |

### What sfmtory actually recovers from these today

Real, current numbers (`sfm run --pipeline incremental`, SIFT, `--pairing exhaustive`,
no `--max-features` cap) — see [Testing with the sample
datasets](../docs/USER_GUIDE.md#testing-with-the-sample-datasets) for the commands and
full discussion:

| | sceaux_castle | temple_sparse_ring | temple_ring |
|---|---|---|---|
| Verified pairs | 41 / 55 | 21 / 120 | 246 / 1081 |
| Images registered | **11 / 11** | **16 / 16** | **47 / 47** |
| 3D points | ~2930 | ~920 | ~6630 |
| Mean reprojection error | 0.42px | 0.37px | **0.30px** |
| Recovered focal length | 2986.7 (true 2905.88, 2.78% off) | 1464.3 (true ~1523.15, 3.86% off) | **1508.1** (true ~1523.15, **0.99%** off) |

For comparison, **real COLMAP** (`pycolmap`, run head-to-head on the exact same photos,
single shared camera per dataset):

| | sceaux_castle | temple_sparse_ring | temple_ring |
|---|---|---|---|
| Images registered | 11 / 11 | 13 / 16 | 47 / 47 |
| Mean reprojection error | 0.62px | 0.22px | 0.32px |
| Recovered focal length | 2973.4 (2.3% off) | 1523.5 (0.02% off) | 1553.6 (2.0% off) |

**sfmtory now matches or beats COLMAP's registered-image count on all three datasets**
(11/11, 16/16, 47/47 vs. COLMAP's 11/11, 13/16, 47/47), closing a gap that was as bad as
5/11 and 5/16 registered earlier in development — see [`decisions.md`](../decisions.md)
for the root causes found and fixed (non-deterministic RANSAC sampling, degenerate-PnP-
sample rejection, nonlinear PnP polish, connected-component-aware multi-seed selection,
a bridge-image bootstrap for chain-shaped match graphs, and SIFT's 2x upsampling for
small images). **On `temple_ring` - the largest, best-conditioned dataset - sfmtory
beats COLMAP outright on all three metrics**: reprojection error (0.30px vs. 0.32px)
and focal length error (0.99% vs. 2.0%), on top of the tied registration count. On the
two smaller/harder datasets, focal length accuracy is still somewhat behind COLMAP's
(2.78% vs. 2.3% on `sceaux_castle`, 3.86% vs. 0.02% on `temple_sparse_ring`) - an
outlier-filtering pass in bundle adjustment, switching from numerical to exact analytic
Jacobians (now implemented for every supported camera model, not just the one both
original test datasets use), and track completion (extending existing points' tracks
with new observations from later-registered images, not just creating fresh points)
together closed most of the original gap (`sceaux_castle` was 5.4%; `temple_sparse_ring`
was 6.3%) - the analytic-Jacobian fix was validated against real Ceres Solver output
during development, though sfmtory has no runtime dependency on Ceres, and on that same
comparison the native solver held up *more* reliably deterministic than Ceres itself on
this specific hard problem. See `decisions.md`'s "Track completion", "Analytic
Jacobians", and "Known open gaps" for the full story.
