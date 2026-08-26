//! DISK: a learned, fully-convolutional keypoint detector + dense descriptor
//! network (Tyszkiewicz, Fua & Trulls, "Learning local features with policy
//! gradient", NeurIPS 2020), run through `ort` (ONNX Runtime's Rust bindings)
//! rather than reimplemented natively - unlike SIFT/ORB/ArUco, DISK's weights
//! are the actual detector (there is no meaningful "native Rust
//! reimplementation" of a trained neural network's weights), so this module
//! is a thin inference wrapper, not a from-scratch reimplementation.
//!
//! This is sfmtory's first GPU-capable pipeline stage: `ort` auto-registers
//! the CUDA/TensorRT execution providers when `DiskParams::use_gpu` is set,
//! silently falling back to the CPU execution provider if no GPU/compatible
//! runtime is available (see `with_session`'s doc comment) - matching
//! PLAN.md's "GPU: optional, auto-detected, everywhere it actually pays off"
//! design. Classical SIFT/ORB extraction stays CPU-only deliberately (see
//! their own module docs / PLAN.md) - DISK and a future LightGlue matcher are
//! the only stages where GPU transfer overhead is actually worth paying.
//!
//! ## Licensing (see decisions.md for the full research)
//!
//! Only the DISK-trained weights are used, never SuperPoint's (SuperPoint's
//! original weights are non-commercially licensed) or ALIKED's (weight
//! license unconfirmed) - matching this project's "no research-only weights
//! bundled" rule (PLAN.md). The model is Apache-2.0 licensed end to end:
//! architecture/weights from `cvlab-epfl/disk`, exported to ONNX by
//! `fabio-sim/LightGlue-ONNX` (also Apache-2.0). Nothing is bundled in this
//! repository - `resolve_model_path` downloads the ~4.4MB model file to a
//! per-user cache directory on first use and verifies it against a hardcoded
//! SHA-256 before ever loading it, exactly like the many small-model CLI
//! tools (whisper.cpp, ollama, ...) that follow this same download-and-cache
//! pattern rather than bloating the repository with a binary weight file.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use image::DynamicImage;
use ort::ep::{ExecutionProvider, CPU, CUDA, TensorRT};
use ort::session::Session;
use ort::value::Tensor;

use sfm_core::{Descriptors, FeatureSet, Keypoint, Result, SfmError};

/// Official standalone-extractor release asset from `fabio-sim/LightGlue-ONNX`
/// v1.0.0 (Apache-2.0) - exported from `cvlab-epfl/disk`'s Apache-2.0 weights
/// via `kornia.feature.DISK.from_pretrained("depth")`. Padding to a multiple
/// of 16 (required by DISK's downsampling architecture) is baked into the
/// graph itself, so callers don't need to pad the input image.
const MODEL_URL: &str =
    "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk.onnx";
const MODEL_SHA256: &str = "f02f18e254bd52d978981c715a4e7961f15afaa23290379b9b357f6745df12c4";
const MODEL_CACHE_FILE: &str = "disk.onnx";

#[derive(Debug, Clone)]
pub struct DiskParams {
    pub max_features: Option<usize>,
    /// Registers the CUDA/TensorRT execution providers ahead of CPU when
    /// `true`. `ort` silently falls back to CPU if neither is usable on this
    /// machine (missing driver/runtime libraries, no compatible GPU) - never
    /// a hard failure just because this was requested (see this module's doc
    /// comment and `with_session`).
    pub use_gpu: bool,
    /// Explicit path to a `.onnx` model file, bypassing the download/cache
    /// path entirely. `None` (the default) resolves the model from the local
    /// cache, downloading it on first use - see `resolve_model_path`.
    pub model_path: Option<PathBuf>,
    /// Images whose longer side exceeds this are downsampled before
    /// inference (keypoint coordinates are scaled back to the *original*
    /// resolution afterward - see `detect`). Unlike SIFT's fixed-cost
    /// Gaussian pyramid, DISK is a dense per-pixel CNN: its memory and
    /// compute cost scale with total pixel count with no natural cutoff, and
    /// running it on full-resolution real photos (e.g. `sceaux_castle`'s
    /// 2832x2128 originals) measurably drove one process past 7GB RSS and
    /// climbing on CPU before being killed during this feature's own
    /// validation - a real risk, not a hypothetical one (see decisions.md).
    /// 1600 matches the same cutoff SIFT's own `UPSAMPLE_MAX_MIN_DIM`
    /// already uses elsewhere in this crate, for consistency.
    pub max_image_dim: Option<u32>,
}

impl Default for DiskParams {
    fn default() -> Self {
        DiskParams {
            // Same order of magnitude as SIFT's own default (12000) and
            // COLMAP's `max_num_features` - DISK's own detector doesn't
            // impose a cap in the exported graph, so this is the only limit.
            max_features: Some(8000),
            use_gpu: false,
            model_path: None,
            max_image_dim: Some(1600),
        }
    }
}

/// Resolves the path to a usable DISK ONNX model file: the explicit
/// `model_path` override if given, otherwise a per-user cache directory,
/// downloading the model there on first use.
///
/// Never re-downloads once the correctly-hashed file is already cached -
/// only re-fetches if the cached file is missing or fails the checksum
/// (e.g. a previous download was interrupted).
fn resolve_model_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }

    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| SfmError::Other("could not determine a user cache directory".into()))?
        .join("sfmtory")
        .join("models");
    std::fs::create_dir_all(&cache_dir)?;
    let cached_path = cache_dir.join(MODEL_CACHE_FILE);

    if cached_path.exists() && sha256_file(&cached_path)? == MODEL_SHA256 {
        return Ok(cached_path);
    }

    eprintln!("downloading DISK model (~4.4MB, one-time) from {MODEL_URL} to {cached_path:?}");
    let response = ureq::get(MODEL_URL)
        .call()
        .map_err(|e| SfmError::Other(format!("downloading DISK model: {e}")))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| SfmError::Other(format!("reading DISK model download: {e}")))?;

    let actual = sha256_bytes(&bytes);
    if actual != MODEL_SHA256 {
        return Err(SfmError::Other(format!(
            "DISK model download failed checksum verification: expected {MODEL_SHA256}, got {actual}"
        )));
    }

    // Write to a temp file then rename, so a crash/interrupt mid-download
    // never leaves a corrupt file at `cached_path` that a later run would
    // wrongly trust (the checksum check above only runs on a pre-existing
    // file at startup, not after every write).
    let tmp_path = cached_path.with_extension("onnx.tmp");
    std::fs::write(&tmp_path, &bytes)?;
    std::fs::rename(&tmp_path, &cached_path)?;

    Ok(cached_path)
}

fn sha256_bytes(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

/// One process-wide DISK session, built lazily on first use and reused for
/// every subsequent image - `ort::Session` isn't cheap to create (it loads
/// and optimizes the whole ONNX graph), and `sfm extract` calls this once per
/// input image. Session::run takes `&mut self`, so concurrent calls from
/// `sfm extract`'s parallel extraction pool share this one session behind a
/// `Mutex` - inference itself is serialized, which is the right tradeoff for
/// a GPU-bound model (concurrent CUDA calls from multiple threads don't
/// actually run faster) and is a no-op cost for the CPU execution provider,
/// which already parallelizes internally per call.
///
/// Only the *first* caller's `params` (model path, GPU flag) take effect;
/// later calls reuse the already-built session. This matches how `sfm
/// extract` actually uses it in practice (one detector config per process),
/// and keeping a single global session avoids repeatedly re-registering
/// execution providers and reloading the graph for no benefit.
///
/// The `Option` is deliberately *inside* the `Mutex` rather than using
/// `OnceLock::get_or_init` directly: `sfm extract`'s parallel extraction
/// pool can call `detect` from several threads at once, and building is
/// fallible (model download/session creation) - `OnceLock` alone has no
/// stable fallible-init API, and a naive "check `OnceLock::get()`, build if
/// `None`" outside a lock lets every thread that arrives before the first
/// build finishes see `None` and redundantly build (and immediately
/// discard) its own session. Locking the whole check-and-build sequence
/// under one `Mutex` makes the race impossible: only the first thread to
/// acquire the lock ever builds anything.
static SESSION: OnceLock<Mutex<Option<Session>>> = OnceLock::new();

fn with_session<T>(
    params: &DiskParams,
    f: impl FnOnce(&mut Session) -> Result<T>,
) -> Result<T> {
    let cell = SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = cell
        .lock()
        .map_err(|_| SfmError::Other("DISK ONNX session lock poisoned".into()))?;

    if guard.is_none() {
        let model_path = resolve_model_path(params.model_path.as_deref())?;

        // `ExecutionProvider::is_available` only reports whether ONNX
        // Runtime was *compiled with support* for the EP (always true here
        // for CUDA/TensorRT, since the prebuilt binary bundles them), not
        // whether it can actually initialize on *this* machine (that also
        // needs a working driver + matching CUDA/cuDNN runtime libraries) -
        // so this is a best-effort, not-fully-reliable "does GPU look
        // possible" hint. The authoritative answer is still whichever EP
        // `apply_execution_providers` (in `ort`'s source) actually manages
        // to register below, reported through this crate's own `tracing`
        // integration - enable it with e.g. `RUST_LOG=ort=info` for the
        // full per-EP registration success/failure detail, since `ort` only
        // logs there, not through any value this function can inspect.
        let mut providers = Vec::new();
        let mut requested = vec!["CPU"];
        if params.use_gpu {
            let cuda_compiled = CUDA::default().is_available().unwrap_or(false);
            let trt_compiled = TensorRT::default().is_available().unwrap_or(false);
            eprintln!(
                "DISK: --gpu requested; ONNX Runtime was compiled with CUDA support: {cuda_compiled}, TensorRT support: {trt_compiled} (compiled-in support alone doesn't guarantee a usable GPU driver/runtime is present - registration is still attempted and falls back to CPU automatically either way)"
            );
            providers.push(CUDA::default().build());
            providers.push(TensorRT::default().build());
            requested = vec!["CUDA", "TensorRT", "CPU"];
        }
        providers.push(CPU::default().build());

        let session = Session::builder()
            .map_err(|e| SfmError::Other(format!("building ONNX Runtime session builder: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| SfmError::Other(format!("registering execution providers: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| SfmError::Other(format!("loading DISK model {model_path:?}: {e}")))?;

        eprintln!("DISK: execution providers requested in priority order: {requested:?}");

        *guard = Some(session);
    }

    f(guard.as_mut().expect("just initialized above"))
}

/// Runs DISK on one already-loaded image, returning its keypoints and
/// L2-normalized 128-d dense descriptors as an ordinary `FeatureSet` -
/// downstream code (matching, verification, reconstruction) doesn't know or
/// care that these came from a neural network rather than SIFT/ORB.
pub fn detect(img: &DynamicImage, params: &DiskParams) -> Result<FeatureSet> {
    let (orig_width, orig_height) = (img.width(), img.height());
    // Downsample large images before inference - see `DiskParams::max_image_dim`'s
    // doc comment for why this exists at all. `inverse_scale` maps a keypoint
    // found in the (possibly smaller) inference-resolution image back to
    // *original* pixel coordinates, since every downstream consumer
    // (matching, triangulation, BA) expects keypoints in the original
    // image's coordinate frame, same as every other detector in this crate.
    let longer_side = orig_width.max(orig_height);
    let (rgb, inverse_scale) = match params.max_image_dim {
        Some(max_dim) if longer_side > max_dim => {
            let scale = max_dim as f64 / longer_side as f64;
            let new_w = ((orig_width as f64 * scale).round() as u32).max(1);
            let new_h = ((orig_height as f64 * scale).round() as u32).max(1);
            let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
            (resized.to_rgb8(), longer_side as f32 / max_dim as f32)
        }
        _ => (img.to_rgb8(), 1.0),
    };
    let (width, height) = (rgb.width(), rgb.height());

    // NCHW, [0, 1]-normalized float32, RGB channel order - matches
    // `fabio-sim/LightGlue-ONNX`'s own `numpy_image_to_torch`
    // (`image.transpose(2, 0, 1) / 255.0`), which is what this exact model
    // was exported against. No ImageNet mean/std subtraction.
    let mut data = vec![0f32; 3 * (width as usize) * (height as usize)];
    let plane = (width as usize) * (height as usize);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let idx = (y as usize) * (width as usize) + (x as usize);
        data[idx] = pixel[0] as f32 / 255.0;
        data[plane + idx] = pixel[1] as f32 / 255.0;
        data[2 * plane + idx] = pixel[2] as f32 / 255.0;
    }

    let input = Tensor::from_array((vec![1i64, 3, height as i64, width as i64], data))
        .map_err(|e| SfmError::Other(format!("building DISK input tensor: {e}")))?;

    let (keypoints, descriptors) = with_session(params, |session| {
        let outputs = session
            .run(ort::inputs!["image" => input])
            .map_err(|e| SfmError::Other(format!("running DISK inference: {e}")))?;

        let (_, kp_data) = outputs["keypoints"]
            .try_extract_tensor::<i64>()
            .map_err(|e| SfmError::Other(format!("reading DISK keypoints output: {e}")))?;
        let (_, score_data) = outputs["scores"]
            .try_extract_tensor::<f32>()
            .map_err(|e| SfmError::Other(format!("reading DISK scores output: {e}")))?;
        let (_, desc_data) = outputs["descriptors"]
            .try_extract_tensor::<f32>()
            .map_err(|e| SfmError::Other(format!("reading DISK descriptors output: {e}")))?;

        let n = score_data.len();
        const DESC_DIM: usize = 128;
        let mut keypoints = Vec::with_capacity(n);
        for i in 0..n {
            keypoints.push(Keypoint {
                // Scaled back to the *original* image's coordinate frame -
                // see `inverse_scale`'s docs above (1.0, a no-op, when no
                // downsampling was needed).
                x: kp_data[2 * i] as f32 * inverse_scale,
                y: kp_data[2 * i + 1] as f32 * inverse_scale,
                // DISK is a dense, single-scale detector with no explicit
                // orientation estimation - 0/1.0 matches the same convention
                // ORB/ArUco already use for detectors that don't estimate
                // these (see `sfm-core::Keypoint`'s field docs).
                scale: 1.0,
                angle: 0.0,
                response: score_data[i],
            });
        }
        let descriptors = Descriptors::Float32 {
            dim: DESC_DIM as u32,
            data: desc_data[..n * DESC_DIM].to_vec(),
        };
        Ok((keypoints, descriptors))
    })?;

    let mut feature_set = FeatureSet {
        keypoints,
        descriptors,
    };
    if let Some(max) = params.max_features {
        feature_set.truncate_to_strongest(max);
    }
    Ok(feature_set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// A textured synthetic image (concentric rings + a grid, unlike a flat
    /// color which no detector - learned or classical - should fire on) for
    /// DISK to actually find keypoints on, mirroring `sift`/`orb`'s own
    /// synthetic-image test pattern.
    fn synthetic_scene(w: u32, h: u32) -> DynamicImage {
        let img = RgbImage::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32, y as f32);
            let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
            let r = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
            let ring = ((r / 6.0).sin() * 0.5 + 0.5) * 255.0;
            let grid = if (x / 8) % 2 == (y / 8) % 2 { 40.0 } else { 0.0 };
            let v = (ring + grid).clamp(0.0, 255.0) as u8;
            Rgb([v, v, v])
        });
        DynamicImage::ImageRgb8(img)
    }

    /// Real inference requires network access on first use (to download the
    /// ~4.4MB model into the local cache - see `resolve_model_path`) and is
    /// slow-ish on CPU, so this is a real (not synthetic-only) integration
    /// check rather than a fast unit test - matches this being an *optional*
    /// capability (PLAN.md), not part of the always-available classical
    /// detector set. Skips cleanly (doesn't fail the suite) if the model
    /// can't be resolved at all, e.g. an offline CI environment.
    #[test]
    fn finds_keypoints_on_textured_image() {
        let params = DiskParams::default();
        if let Err(e) = resolve_model_path(params.model_path.as_deref()) {
            eprintln!("skipping DISK test: model not resolvable ({e})");
            return;
        }

        let img = synthetic_scene(256, 256);
        let features = detect(&img, &params).expect("DISK inference should succeed");
        assert!(
            !features.is_empty(),
            "expected DISK to find keypoints on a textured image"
        );
        assert_eq!(features.descriptors.len(), features.keypoints.len());
        match &features.descriptors {
            Descriptors::Float32 { dim, .. } => assert_eq!(*dim, 128),
            _ => panic!("expected float descriptors"),
        }

        // Descriptors are L2-normalized by the exported graph itself (see
        // this module's doc comment) - a real invariant of DISK's output,
        // not something sfmtory computes, so worth asserting on directly.
        for i in 0..features.len() {
            let row = features.descriptors.float_row(i).unwrap();
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-2,
                "descriptor {i} not unit-normalized: {norm}"
            );
        }
    }

    #[test]
    fn max_features_truncates_to_strongest() {
        let mut params = DiskParams::default();
        if let Err(e) = resolve_model_path(params.model_path.as_deref()) {
            eprintln!("skipping DISK test: model not resolvable ({e})");
            return;
        }
        params.max_features = Some(10);

        let img = synthetic_scene(256, 256);
        let features = detect(&img, &params).expect("DISK inference should succeed");
        assert!(features.len() <= 10);
    }
}
