//! Detector-agnostic feature types shared by `sfm-features` (which produces
//! them), `sfm-match` (which consumes them), and the project database.

use serde::{Deserialize, Serialize};

/// Bytes per fiducial corner descriptor: `capture_id`, `marker_id`,
/// `corner_index`, each a little-endian `u32`.
pub const MARKER_CORNER_BYTES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Characteristic scale (SIFT: `sigma` at detection octave; ORB/ArUco: 1.0).
    pub scale: f32,
    /// Dominant orientation in radians, used to make descriptors
    /// rotation-invariant. 0 for detectors that don't estimate one.
    pub angle: f32,
    /// Detector-specific strength/quality score, used to cap `--max-features`
    /// by keeping the strongest keypoints.
    pub response: f32,
}

/// Descriptor storage for a whole image, kept flat (not `Vec<Vec<_>>`) so it
/// serializes compactly into the project database as one BLOB per image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Descriptors {
    /// Row-major `[num_keypoints * dim]`, e.g. SIFT's 128-d float descriptor.
    Float32 { dim: u32, data: Vec<f32> },
    /// Row-major `[num_keypoints * bytes_per_descriptor]` packed bits, e.g.
    /// ORB's 256-bit (32-byte) BRIEF descriptor, compared via Hamming distance.
    Binary {
        bytes_per_descriptor: u32,
        data: Vec<u8>,
    },
    /// ArUco/fiducial marker corners: 12 bytes per keypoint, three LE `u32`s -
    /// `capture_id`, `marker_id`, `corner_index` (0..3, clockwise from
    /// top-left). Matched by exact equality of the whole triple rather than
    /// nearest-neighbour search (see `sfm-match`).
    ///
    /// `capture_id` is part of the identity, not decoration: a fiducial that
    /// is physically moved between captures is a *different 3D point* in each
    /// one, and matching marker 5 in capture 0 to marker 5 in capture 1 would
    /// silently fabricate a correspondence between two unrelated locations.
    /// Including the capture in the key makes that impossible. Datasets with
    /// no captures use `capture_id = 0` throughout, so the behaviour is
    /// unchanged for them.
    MarkerCorner { data: Vec<u8> },
}

impl Descriptors {
    pub fn len(&self) -> usize {
        match self {
            Descriptors::Float32 { dim, data } => {
                if *dim == 0 {
                    0
                } else {
                    data.len() / *dim as usize
                }
            }
            Descriptors::Binary {
                bytes_per_descriptor,
                data,
            } => {
                if *bytes_per_descriptor == 0 {
                    0
                } else {
                    data.len() / *bytes_per_descriptor as usize
                }
            }
            Descriptors::MarkerCorner { data } => data.len() / MARKER_CORNER_BYTES,
        }
    }

    pub fn float_row(&self, i: usize) -> Option<&[f32]> {
        match self {
            Descriptors::Float32 { dim, data } => {
                let dim = *dim as usize;
                Some(&data[i * dim..(i + 1) * dim])
            }
            _ => None,
        }
    }

    pub fn binary_row(&self, i: usize) -> Option<&[u8]> {
        match self {
            Descriptors::Binary {
                bytes_per_descriptor,
                data,
            } => {
                let n = *bytes_per_descriptor as usize;
                Some(&data[i * n..(i + 1) * n])
            }
            _ => None,
        }
    }

    /// `(capture_id, marker_id, corner_index)` - the full match key.
    pub fn marker_corner(&self, i: usize) -> Option<(u32, u32, u32)> {
        match self {
            Descriptors::MarkerCorner { data } => {
                let row = &data[i * MARKER_CORNER_BYTES..(i + 1) * MARKER_CORNER_BYTES];
                let capture_id = u32::from_le_bytes(row[0..4].try_into().unwrap());
                let marker_id = u32::from_le_bytes(row[4..8].try_into().unwrap());
                let corner_index = u32::from_le_bytes(row[8..12].try_into().unwrap());
                Some((capture_id, marker_id, corner_index))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub keypoints: Vec<Keypoint>,
    pub descriptors: Descriptors,
}

impl FeatureSet {
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    /// Keep only the `max` highest-`response` keypoints (and their matching
    /// descriptor rows). No-op if already within budget.
    pub fn truncate_to_strongest(&mut self, max: usize) {
        if self.keypoints.len() <= max {
            return;
        }
        let mut order: Vec<usize> = (0..self.keypoints.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            self.keypoints[b]
                .response
                .partial_cmp(&self.keypoints[a].response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(max);
        order.sort_unstable();

        let new_keypoints: Vec<Keypoint> = order.iter().map(|&i| self.keypoints[i]).collect();
        let new_descriptors = match &self.descriptors {
            Descriptors::Float32 { dim, data } => {
                let dim = *dim as usize;
                let mut new_data = Vec::with_capacity(order.len() * dim);
                for &i in &order {
                    new_data.extend_from_slice(&data[i * dim..(i + 1) * dim]);
                }
                Descriptors::Float32 {
                    dim: dim as u32,
                    data: new_data,
                }
            }
            Descriptors::Binary {
                bytes_per_descriptor,
                data,
            } => {
                let n = *bytes_per_descriptor as usize;
                let mut new_data = Vec::with_capacity(order.len() * n);
                for &i in &order {
                    new_data.extend_from_slice(&data[i * n..(i + 1) * n]);
                }
                Descriptors::Binary {
                    bytes_per_descriptor: n as u32,
                    data: new_data,
                }
            }
            Descriptors::MarkerCorner { data } => {
                let mut new_data = Vec::with_capacity(order.len() * MARKER_CORNER_BYTES);
                for &i in &order {
                    new_data.extend_from_slice(
                        &data[i * MARKER_CORNER_BYTES..(i + 1) * MARKER_CORNER_BYTES],
                    );
                }
                Descriptors::MarkerCorner { data: new_data }
            }
        };
        self.keypoints = new_keypoints;
        self.descriptors = new_descriptors;
    }
}
