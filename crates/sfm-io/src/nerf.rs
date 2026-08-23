//! Reader/writer for the NeRF-ecosystem `transforms.json` convention used by
//! instant-ngp / nerfstudio (see nerfstudio's "Data conventions" docs).
//!
//! Camera-to-world matrices use the OpenGL/Blender axis convention (+X right,
//! +Y up, +Z backward), which is why every pose here is converted through
//! [`sfm_core::pose::Pose::to_nerf_c2w`] / `from_nerf_c2w` rather than written
//! as COLMAP's raw world-to-camera quaternion.

use std::fs;
use std::path::Path;

use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use sfm_core::error::{Result, SfmError};
use sfm_core::pose::Pose;
use sfm_core::{Camera, CameraModel, Image, Reconstruction};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerfFrame {
    pub file_path: String,
    pub transform_matrix: [[f64; 4]; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerfTransforms {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2: Option<f64>,
    pub camera_model: String,
    pub frames: Vec<NerfFrame>,
}

/// Build a `transforms.json` payload from a `Reconstruction`. When every image
/// shares one camera (the common single-camera-capture case) intrinsics are
/// written once at the top level; otherwise they're written per-frame, which
/// nerfstudio also supports.
pub fn to_nerf_transforms(recon: &Reconstruction) -> NerfTransforms {
    let single_camera = recon.cameras.len() == 1;
    let shared: Option<&Camera> = if single_camera {
        recon.cameras.values().next()
    } else {
        None
    };

    let (top_fx, top_fy, top_cx, top_cy, top_w, top_h, top_k1, top_k2, top_p1, top_p2, model_name) =
        if let Some(cam) = shared {
            let (fx, fy) = cam.model.focal_lengths();
            let (cx, cy) = cam.model.principal_point();
            let [k1, k2, p1, p2, _k3, _k4] = cam.model.opencv_distortion();
            (
                Some(fx),
                Some(fy),
                Some(cx),
                Some(cy),
                Some(cam.width as f64),
                Some(cam.height as f64),
                Some(k1),
                Some(k2),
                Some(p1),
                Some(p2),
                "OPENCV".to_string(),
            )
        } else {
            (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "OPENCV".to_string(),
            )
        };

    let mut frames = Vec::with_capacity(recon.images.len());
    for img in recon.images.values() {
        let c2w = img.pose.to_nerf_c2w();
        let m = matrix4_to_array(&c2w);
        let per_frame_intrinsics = shared.is_none();
        let (fl_x, fl_y, cx, cy, w, h) = if per_frame_intrinsics {
            if let Some(cam) = recon.cameras.get(&img.camera_id) {
                let (fx, fy) = cam.model.focal_lengths();
                let (cx, cy) = cam.model.principal_point();
                (
                    Some(fx),
                    Some(fy),
                    Some(cx),
                    Some(cy),
                    Some(cam.width as f64),
                    Some(cam.height as f64),
                )
            } else {
                (None, None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None, None)
        };
        frames.push(NerfFrame {
            file_path: img.name.clone(),
            transform_matrix: m,
            fl_x,
            fl_y,
            cx,
            cy,
            w,
            h,
        });
    }

    NerfTransforms {
        fl_x: top_fx,
        fl_y: top_fy,
        cx: top_cx,
        cy: top_cy,
        w: top_w,
        h: top_h,
        k1: top_k1,
        k2: top_k2,
        p1: top_p1,
        p2: top_p2,
        camera_model: model_name,
        frames,
    }
}

pub fn write_transforms(recon: &Reconstruction, path: &Path) -> Result<()> {
    let transforms = to_nerf_transforms(recon);
    let json = serde_json::to_string_pretty(&transforms)
        .map_err(|e| SfmError::Other(format!("serializing transforms.json: {e}")))?;
    fs::write(path, json)?;
    Ok(())
}

/// Parse a `transforms.json` back into a `Reconstruction`. Since the format
/// doesn't carry a 3D point cloud, `points3d` is left empty; this is primarily
/// useful for round-trip testing and for consuming NeRF-format camera poses
/// produced by other tools as a starting point.
pub fn read_transforms(path: &Path) -> Result<Reconstruction> {
    let content = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&content)
        .map_err(|e| SfmError::Other(format!("parsing transforms.json: {e}")))?;

    let top_f = |key: &str| value.get(key).and_then(Value::as_f64);
    let top_fl_x = top_f("fl_x");
    let top_fl_y = top_f("fl_y").or(top_fl_x);
    let top_cx = top_f("cx");
    let top_cy = top_f("cy");
    let top_w = top_f("w");
    let top_h = top_f("h");
    let top_k1 = top_f("k1").unwrap_or(0.0);
    let top_k2 = top_f("k2").unwrap_or(0.0);
    let top_p1 = top_f("p1").unwrap_or(0.0);
    let top_p2 = top_f("p2").unwrap_or(0.0);
    // camera_angle_x -> fl_x, if focal length wasn't given directly.
    let angle_x = top_f("camera_angle_x");

    let frames = value
        .get("frames")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut recon = Reconstruction::new();
    let mut next_camera_id = 1u32;
    let mut camera_of_key: std::collections::HashMap<(u64, u64, u64, u64), u32> =
        std::collections::HashMap::new();

    for (idx, frame) in frames.iter().enumerate() {
        let file_path = frame
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| SfmError::Other(format!("frame {idx} missing file_path")))?
            .to_string();

        let fget = |key: &str| frame.get(key).and_then(Value::as_f64);
        let w = fget("w").or(top_w).unwrap_or(0.0);
        let h = fget("h").or(top_h).unwrap_or(0.0);
        let fl_x = fget("fl_x")
            .or(top_fl_x)
            .or_else(|| angle_x.map(|a| 0.5 * w / (0.5 * a).tan()))
            .ok_or_else(|| SfmError::Other(format!("frame {idx}: no focal length available")))?;
        let fl_y = fget("fl_y").or(top_fl_y).unwrap_or(fl_x);
        let cx = fget("cx").or(top_cx).unwrap_or(w / 2.0);
        let cy = fget("cy").or(top_cy).unwrap_or(h / 2.0);
        let k1 = fget("k1").unwrap_or(top_k1);
        let k2 = fget("k2").unwrap_or(top_k2);
        let p1 = fget("p1").unwrap_or(top_p1);
        let p2 = fget("p2").unwrap_or(top_p2);

        let key = (fl_x.to_bits(), fl_y.to_bits(), w.to_bits(), h.to_bits());
        let camera_id = *camera_of_key.entry(key).or_insert_with(|| {
            let id = next_camera_id;
            next_camera_id += 1;
            recon.cameras.insert(
                id,
                Camera {
                    camera_id: id,
                    model: CameraModel::OpenCV {
                        fx: fl_x,
                        fy: fl_y,
                        cx,
                        cy,
                        k1,
                        k2,
                        p1,
                        p2,
                    },
                    width: w as u32,
                    height: h as u32,
                },
            );
            id
        });

        let tm = frame
            .get("transform_matrix")
            .and_then(Value::as_array)
            .ok_or_else(|| SfmError::Other(format!("frame {idx} missing transform_matrix")))?;
        let mut m = Matrix4::identity();
        for r in 0..4 {
            let row = tm[r].as_array().ok_or_else(|| {
                SfmError::Other(format!("frame {idx}: bad transform_matrix row {r}"))
            })?;
            for c in 0..4 {
                m[(r, c)] = row[c].as_f64().unwrap_or(0.0);
            }
        }
        let pose = Pose::from_nerf_c2w(&m);

        let image_id = (idx + 1) as u32;
        let mut image = Image::new_unregistered(image_id, camera_id, file_path);
        image.pose = pose;
        recon.images.insert(image_id, image);
    }

    Ok(recon)
}

fn matrix4_to_array(m: &Matrix4<f64>) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use sfm_core::CameraModel;

    fn sample_recon() -> Reconstruction {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: CameraModel::OpenCV {
                    fx: 1200.0,
                    fy: 1200.0,
                    cx: 640.0,
                    cy: 360.0,
                    k1: -0.02,
                    k2: 0.01,
                    p1: 0.0,
                    p2: 0.0,
                },
                width: 1280,
                height: 720,
            },
        );
        let mut img = Image::new_unregistered(1, 1, "frame_0001.png".to_string());
        img.pose = Pose::from_quaternion_wxyz_translation(
            [0.95, 0.1, -0.2, 0.05],
            Vector3::new(0.3, 0.1, 2.0),
        );
        recon.images.insert(1, img);
        recon
    }

    #[test]
    fn pose_round_trips_through_nerf_c2w() {
        let recon = sample_recon();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transforms.json");
        write_transforms(&recon, &path).unwrap();
        let read_back = read_transforms(&path).unwrap();

        let orig = &recon.images[&1].pose;
        let got = &read_back.images[&1].pose;

        let orig_center = orig.camera_center();
        let got_center = got.camera_center();
        assert!((orig_center - got_center).norm() < 1e-9);

        // Rotation should match up to the stored representation (both encode
        // the same world-to-camera transform).
        let p_world = Vector3::new(1.0, 2.0, 3.0);
        let a = orig.transform_point(&p_world);
        let b = got.transform_point(&p_world);
        assert!((a - b).norm() < 1e-9);

        let cam = &read_back.cameras[&1];
        assert_eq!(cam.width, 1280);
        assert_eq!(cam.height, 720);
    }
}
