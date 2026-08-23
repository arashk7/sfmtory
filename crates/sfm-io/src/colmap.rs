//! Reader/writer for COLMAP's plain-text sparse model
//! (`cameras.txt` / `images.txt` / `points3D.txt`), as documented at
//! <https://colmap.github.io/format.html>. Round-trips losslessly with our
//! own `Reconstruction` type so results are drop-in usable by COLMAP, Meshlab,
//! Blender's COLMAP importer, nerfstudio, etc.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use nalgebra::Vector3;
use sfm_core::error::{Result, SfmError};
use sfm_core::pose::Pose;
use sfm_core::{Camera, CameraModel, Image, Point3D, Reconstruction, TrackElement};

fn parse_err(context: &str, message: impl Into<String>) -> SfmError {
    SfmError::Parse {
        context: context.to_string(),
        message: message.into(),
    }
}

/// Write `cameras.txt`, `images.txt`, `points3D.txt` into `dir` (created if
/// missing).
pub fn write_model(recon: &Reconstruction, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;
    write_cameras(recon, &dir.join("cameras.txt"))?;
    write_images(recon, &dir.join("images.txt"))?;
    write_points3d(recon, &dir.join("points3D.txt"))?;
    Ok(())
}

/// Read a full model from a directory containing the three COLMAP `.txt` files.
pub fn read_model(dir: &Path) -> Result<Reconstruction> {
    let cameras = read_cameras(&dir.join("cameras.txt"))?;
    let mut recon = Reconstruction {
        cameras,
        ..Default::default()
    };
    recon.images = read_images(&dir.join("images.txt"))?;
    recon.points3d = read_points3d(&dir.join("points3D.txt"))?;
    Ok(recon)
}

fn write_cameras(recon: &Reconstruction, path: &Path) -> Result<()> {
    let mut out = String::new();
    out.push_str("# Camera list with one line of data per camera:\n");
    out.push_str("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    out.push_str(&format!("# Number of cameras: {}\n", recon.cameras.len()));
    for cam in recon.cameras.values() {
        let params: Vec<String> = cam
            .model
            .params()
            .iter()
            .map(|p| format!("{p:.10}"))
            .collect();
        out.push_str(&format!(
            "{} {} {} {} {}\n",
            cam.camera_id,
            cam.model.name(),
            cam.width,
            cam.height,
            params.join(" ")
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn read_cameras(path: &Path) -> Result<BTreeMap<u32, Camera>> {
    let content = fs::read_to_string(path)?;
    let mut cameras = BTreeMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            return Err(parse_err("cameras.txt", format!("malformed line: {line}")));
        }
        let camera_id: u32 = tokens[0]
            .parse()
            .map_err(|_| parse_err("cameras.txt", "bad CAMERA_ID"))?;
        let model_name = tokens[1];
        let width: u32 = tokens[2]
            .parse()
            .map_err(|_| parse_err("cameras.txt", "bad WIDTH"))?;
        let height: u32 = tokens[3]
            .parse()
            .map_err(|_| parse_err("cameras.txt", "bad HEIGHT"))?;
        let params: Vec<f64> = tokens[4..]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| parse_err("cameras.txt", "bad PARAMS"))?;
        let model = CameraModel::from_name_and_params(model_name, &params)
            .ok_or_else(|| SfmError::UnknownCameraModel(model_name.to_string()))?;
        cameras.insert(
            camera_id,
            Camera {
                camera_id,
                model,
                width,
                height,
            },
        );
    }
    Ok(cameras)
}

fn write_images(recon: &Reconstruction, path: &Path) -> Result<()> {
    let total_points2d: usize = recon.images.values().map(|im| im.keypoints.len()).sum();
    let mean_obs = if recon.images.is_empty() {
        0.0
    } else {
        total_points2d as f64 / recon.images.len() as f64
    };
    let mut out = String::new();
    out.push_str("# Image list with two lines of data per image:\n");
    out.push_str("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
    out.push_str("#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
    out.push_str(&format!(
        "# Number of images: {}, mean observations per image: {mean_obs:.1}\n",
        recon.images.len()
    ));
    for img in recon.images.values() {
        let [qw, qx, qy, qz] = img.pose.quaternion_wxyz();
        let t = img.pose.translation;
        out.push_str(&format!(
            "{} {qw:.10} {qx:.10} {qy:.10} {qz:.10} {:.10} {:.10} {:.10} {} {}\n",
            img.id, t.x, t.y, t.z, img.camera_id, img.name
        ));
        let points_line: Vec<String> = img
            .keypoints
            .iter()
            .zip(img.point3d_ids.iter())
            .map(|((x, y), pid)| {
                let pid = pid.map(|p| p as i64).unwrap_or(-1);
                format!("{x:.3} {y:.3} {pid}")
            })
            .collect();
        out.push_str(&points_line.join(" "));
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

fn read_images(path: &Path) -> Result<BTreeMap<u32, Image>> {
    let content = fs::read_to_string(path)?;
    let mut images = BTreeMap::new();
    let mut lines = content.lines().filter(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#')
    });
    while let Some(header) = lines.next() {
        let tokens: Vec<&str> = header.split_whitespace().collect();
        if tokens.len() < 10 {
            return Err(parse_err(
                "images.txt",
                format!("malformed header: {header}"),
            ));
        }
        let id: u32 = tokens[0]
            .parse()
            .map_err(|_| parse_err("images.txt", "bad IMAGE_ID"))?;
        let q: Vec<f64> = tokens[1..5]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| parse_err("images.txt", "bad quaternion"))?;
        let t: Vec<f64> = tokens[5..8]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| parse_err("images.txt", "bad translation"))?;
        let camera_id: u32 = tokens[8]
            .parse()
            .map_err(|_| parse_err("images.txt", "bad CAMERA_ID"))?;
        let name = tokens[9].to_string();
        let pose = Pose::from_quaternion_wxyz_translation(
            [q[0], q[1], q[2], q[3]],
            Vector3::new(t[0], t[1], t[2]),
        );

        let points_line = lines
            .next()
            .ok_or_else(|| parse_err("images.txt", "missing POINTS2D line"))?;
        let ptoks: Vec<&str> = points_line.split_whitespace().collect();
        let mut keypoints = Vec::with_capacity(ptoks.len() / 3);
        let mut point3d_ids = Vec::with_capacity(ptoks.len() / 3);
        for chunk in ptoks.chunks(3) {
            if chunk.len() < 3 {
                break;
            }
            let x: f32 = chunk[0]
                .parse()
                .map_err(|_| parse_err("images.txt", "bad point x"))?;
            let y: f32 = chunk[1]
                .parse()
                .map_err(|_| parse_err("images.txt", "bad point y"))?;
            let pid: i64 = chunk[2]
                .parse()
                .map_err(|_| parse_err("images.txt", "bad POINT3D_ID"))?;
            keypoints.push((x, y));
            point3d_ids.push(if pid < 0 { None } else { Some(pid as u64) });
        }

        images.insert(
            id,
            Image {
                id,
                camera_id,
                name,
                pose,
                keypoints,
                point3d_ids,
            },
        );
    }
    Ok(images)
}

fn write_points3d(recon: &Reconstruction, path: &Path) -> Result<()> {
    let mut out = String::new();
    out.push_str("# 3D point list with one line of data per point:\n");
    out.push_str("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    out.push_str(&format!("# Number of points: {}\n", recon.points3d.len()));
    for p in recon.points3d.values() {
        let track: Vec<String> = p
            .track
            .iter()
            .map(|te| format!("{} {}", te.image_id, te.point2d_idx))
            .collect();
        out.push_str(&format!(
            "{} {:.10} {:.10} {:.10} {} {} {} {:.6} {}\n",
            p.id,
            p.xyz.x,
            p.xyz.y,
            p.xyz.z,
            p.color[0],
            p.color[1],
            p.color[2],
            p.error,
            track.join(" ")
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn read_points3d(path: &Path) -> Result<BTreeMap<u64, Point3D>> {
    let content = fs::read_to_string(path)?;
    let mut points = BTreeMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 8 {
            return Err(parse_err("points3D.txt", format!("malformed line: {line}")));
        }
        let id: u64 = tokens[0]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad POINT3D_ID"))?;
        let x: f64 = tokens[1]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad X"))?;
        let y: f64 = tokens[2]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad Y"))?;
        let z: f64 = tokens[3]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad Z"))?;
        let r: u8 = tokens[4]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad R"))?;
        let g: u8 = tokens[5]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad G"))?;
        let b: u8 = tokens[6]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad B"))?;
        let error: f64 = tokens[7]
            .parse()
            .map_err(|_| parse_err("points3D.txt", "bad ERROR"))?;
        let mut track = Vec::new();
        for chunk in tokens[8..].chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let image_id: u32 = chunk[0]
                .parse()
                .map_err(|_| parse_err("points3D.txt", "bad TRACK IMAGE_ID"))?;
            let point2d_idx: u32 = chunk[1]
                .parse()
                .map_err(|_| parse_err("points3D.txt", "bad TRACK POINT2D_IDX"))?;
            track.push(TrackElement {
                image_id,
                point2d_idx,
            });
        }
        points.insert(
            id,
            Point3D {
                id,
                xyz: Vector3::new(x, y, z),
                color: [r, g, b],
                error,
                track,
            },
        );
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfm_core::model::TrackElement as TE;

    fn sample_recon() -> Reconstruction {
        let mut recon = Reconstruction::new();
        recon.cameras.insert(
            1,
            Camera {
                camera_id: 1,
                model: CameraModel::SimpleRadial {
                    f: 1600.0,
                    cx: 960.0,
                    cy: 540.0,
                    k: -0.05,
                },
                width: 1920,
                height: 1080,
            },
        );
        let mut img = Image::new_unregistered(1, 1, "img_0001.jpg".to_string());
        img.keypoints = vec![(100.5, 200.25), (300.0, 400.0)];
        img.point3d_ids = vec![Some(1), None];
        img.pose = Pose::from_quaternion_wxyz_translation(
            [0.98, 0.1, 0.05, 0.02],
            Vector3::new(0.1, -0.2, 1.5),
        );
        recon.images.insert(1, img);
        recon.points3d.insert(
            1,
            Point3D {
                id: 1,
                xyz: Vector3::new(1.0, 2.0, 3.0),
                color: [255, 128, 0],
                error: 0.42,
                track: vec![TE {
                    image_id: 1,
                    point2d_idx: 0,
                }],
            },
        );
        recon
    }

    #[test]
    fn round_trips_through_text_files() {
        let recon = sample_recon();
        let dir = tempfile::tempdir().unwrap();
        write_model(&recon, dir.path()).unwrap();
        let read_back = read_model(dir.path()).unwrap();

        assert_eq!(read_back.cameras.len(), 1);
        assert_eq!(read_back.images.len(), 1);
        assert_eq!(read_back.points3d.len(), 1);

        let cam = &read_back.cameras[&1];
        assert_eq!(cam.model.name(), "SIMPLE_RADIAL");
        assert_eq!(cam.width, 1920);

        let img = &read_back.images[&1];
        assert_eq!(img.name, "img_0001.jpg");
        assert_eq!(img.keypoints.len(), 2);
        assert_eq!(img.point3d_ids, vec![Some(1), None]);

        let p = &read_back.points3d[&1];
        assert_eq!(p.color, [255, 128, 0]);
        assert_eq!(p.track.len(), 1);
        assert!((p.xyz.x - 1.0).abs() < 1e-6);
    }
}
