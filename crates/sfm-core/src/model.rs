//! In-memory sparse reconstruction model: cameras, registered images (with
//! their 2D keypoints and pose), and triangulated 3D points with tracks.
//! Mirrors COLMAP's `cameras.txt`/`images.txt`/`points3D.txt` triple so it can
//! be losslessly exported to that format (see `sfm-io`).

use std::collections::BTreeMap;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::camera::Camera;
use crate::pose::Pose;

/// One observation of a 3D point in one image's keypoint list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackElement {
    pub image_id: u32,
    /// Index into that image's `keypoints`/`point3d_ids` arrays.
    pub point2d_idx: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub id: u64,
    pub xyz: Vector3<f64>,
    pub color: [u8; 3],
    /// Mean reprojection error in pixels across the track, as of the last BA.
    pub error: f64,
    pub track: Vec<TrackElement>,
}

/// A registered image: its pose and the 2D keypoints detected in it, each
/// optionally linked to a triangulated `Point3D` (`point3d_ids[i] == None`
/// means keypoint `i` has no accepted 3D correspondence yet).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub id: u32,
    pub camera_id: u32,
    pub name: String,
    pub pose: Pose,
    /// Pixel coordinates (x, y) of every detected keypoint, index-aligned with
    /// `point3d_ids`.
    pub keypoints: Vec<(f32, f32)>,
    pub point3d_ids: Vec<Option<u64>>,
}

impl Image {
    pub fn new_unregistered(id: u32, camera_id: u32, name: String) -> Self {
        Image {
            id,
            camera_id,
            name,
            pose: Pose::identity(),
            keypoints: Vec::new(),
            point3d_ids: Vec::new(),
        }
    }
}

/// A full sparse reconstruction: the output of the `map`/`refine` stages, and
/// the input/output of `export`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Reconstruction {
    pub cameras: BTreeMap<u32, Camera>,
    pub images: BTreeMap<u32, Image>,
    pub points3d: BTreeMap<u64, Point3D>,
}

impl Reconstruction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_registered_images(&self) -> usize {
        self.images.len()
    }

    pub fn next_point3d_id(&self) -> u64 {
        self.points3d.keys().next_back().map_or(1, |id| id + 1)
    }

    /// Mean reprojection error (pixels) across all points' `error` field,
    /// weighted by track length. Used by `sfm eval` and BA convergence logs.
    pub fn mean_reprojection_error(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0.0;
        for p in self.points3d.values() {
            let n = p.track.len() as f64;
            total += p.error * n;
            count += n;
        }
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }
}
