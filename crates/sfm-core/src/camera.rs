//! Camera intrinsic models, compatible with COLMAP's naming and parameter order
//! so exported models are drop-in readable by COLMAP/other tools.

use serde::{Deserialize, Serialize};

/// One physical camera's intrinsics. Multiple `Image`s may share a `Camera`
/// (same `camera_id`) when captured by the same physical sensor/lens, which is
/// what lets calibration pool observations across many images of one camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub camera_id: u32,
    pub model: CameraModel,
    pub width: u32,
    pub height: u32,
}

/// Supported intrinsic/distortion models. Variants and parameter order match
/// COLMAP exactly (see `doc/format.rst` / `src/colmap/sensor/models.h`) so that
/// `params()` round-trips through `cameras.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CameraModel {
    /// params: f, cx, cy
    SimplePinhole { f: f64, cx: f64, cy: f64 },
    /// params: fx, fy, cx, cy
    Pinhole { fx: f64, fy: f64, cx: f64, cy: f64 },
    /// params: f, cx, cy, k
    SimpleRadial { f: f64, cx: f64, cy: f64, k: f64 },
    /// params: f, cx, cy, k1, k2
    Radial {
        f: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
    },
    /// params: fx, fy, cx, cy, k1, k2, p1, p2
    OpenCV {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
    },
    /// params: fx, fy, cx, cy, k1, k2, k3, k4 (equidistant fisheye)
    OpenCVFisheye {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    },
}

impl CameraModel {
    /// COLMAP's model name string, as it appears in `cameras.txt`.
    pub fn name(&self) -> &'static str {
        match self {
            CameraModel::SimplePinhole { .. } => "SIMPLE_PINHOLE",
            CameraModel::Pinhole { .. } => "PINHOLE",
            CameraModel::SimpleRadial { .. } => "SIMPLE_RADIAL",
            CameraModel::Radial { .. } => "RADIAL",
            CameraModel::OpenCV { .. } => "OPENCV",
            CameraModel::OpenCVFisheye { .. } => "OPENCV_FISHEYE",
        }
    }

    /// COLMAP's numeric model id, as used internally and in the binary format.
    pub fn model_id(&self) -> i32 {
        match self {
            CameraModel::SimplePinhole { .. } => 0,
            CameraModel::Pinhole { .. } => 1,
            CameraModel::SimpleRadial { .. } => 2,
            CameraModel::Radial { .. } => 3,
            CameraModel::OpenCV { .. } => 4,
            CameraModel::OpenCVFisheye { .. } => 5,
        }
    }

    pub fn from_name_and_params(name: &str, params: &[f64]) -> Option<Self> {
        Some(match name {
            "SIMPLE_PINHOLE" => CameraModel::SimplePinhole {
                f: params[0],
                cx: params[1],
                cy: params[2],
            },
            "PINHOLE" => CameraModel::Pinhole {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            "SIMPLE_RADIAL" => CameraModel::SimpleRadial {
                f: params[0],
                cx: params[1],
                cy: params[2],
                k: params[3],
            },
            "RADIAL" => CameraModel::Radial {
                f: params[0],
                cx: params[1],
                cy: params[2],
                k1: params[3],
                k2: params[4],
            },
            "OPENCV" => CameraModel::OpenCV {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
                k1: params[4],
                k2: params[5],
                p1: params[6],
                p2: params[7],
            },
            "OPENCV_FISHEYE" => CameraModel::OpenCVFisheye {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
                k1: params[4],
                k2: params[5],
                k3: params[6],
                k4: params[7],
            },
            _ => return None,
        })
    }

    /// Flat parameter vector in COLMAP order (as written to `cameras.txt`).
    pub fn params(&self) -> Vec<f64> {
        match *self {
            CameraModel::SimplePinhole { f, cx, cy } => vec![f, cx, cy],
            CameraModel::Pinhole { fx, fy, cx, cy } => vec![fx, fy, cx, cy],
            CameraModel::SimpleRadial { f, cx, cy, k } => vec![f, cx, cy, k],
            CameraModel::Radial { f, cx, cy, k1, k2 } => vec![f, cx, cy, k1, k2],
            CameraModel::OpenCV {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            } => vec![fx, fy, cx, cy, k1, k2, p1, p2],
            CameraModel::OpenCVFisheye {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            } => vec![fx, fy, cx, cy, k1, k2, k3, k4],
        }
    }

    /// Focal lengths (fx, fy), useful for NeRF-style export.
    pub fn focal_lengths(&self) -> (f64, f64) {
        match *self {
            CameraModel::SimplePinhole { f, .. } => (f, f),
            CameraModel::Pinhole { fx, fy, .. } => (fx, fy),
            CameraModel::SimpleRadial { f, .. } => (f, f),
            CameraModel::Radial { f, .. } => (f, f),
            CameraModel::OpenCV { fx, fy, .. } => (fx, fy),
            CameraModel::OpenCVFisheye { fx, fy, .. } => (fx, fy),
        }
    }

    /// Principal point (cx, cy).
    pub fn principal_point(&self) -> (f64, f64) {
        match *self {
            CameraModel::SimplePinhole { cx, cy, .. }
            | CameraModel::Pinhole { cx, cy, .. }
            | CameraModel::SimpleRadial { cx, cy, .. }
            | CameraModel::Radial { cx, cy, .. }
            | CameraModel::OpenCV { cx, cy, .. }
            | CameraModel::OpenCVFisheye { cx, cy, .. } => (cx, cy),
        }
    }

    /// Radial/tangential distortion coefficients in OpenCV's (k1, k2, p1, p2,
    /// k3, k4) slots, zero-filled for models that don't have them. Used by the
    /// NeRF/`transforms.json` exporter, which always writes the OpenCV slots.
    pub fn opencv_distortion(&self) -> [f64; 6] {
        match *self {
            CameraModel::SimplePinhole { .. } | CameraModel::Pinhole { .. } => [0.0; 6],
            CameraModel::SimpleRadial { k, .. } => [k, 0.0, 0.0, 0.0, 0.0, 0.0],
            CameraModel::Radial { k1, k2, .. } => [k1, k2, 0.0, 0.0, 0.0, 0.0],
            CameraModel::OpenCV { k1, k2, p1, p2, .. } => [k1, k2, p1, p2, 0.0, 0.0],
            CameraModel::OpenCVFisheye { k1, k2, k3, k4, .. } => [k1, k2, 0.0, 0.0, k3, k4],
        }
    }

    /// Project a 3D point already in camera space (X, Y, Z with Z > 0 forward)
    /// to pixel coordinates, applying this model's distortion.
    pub fn project(&self, p_camera: &nalgebra::Vector3<f64>) -> (f64, f64) {
        let x = p_camera.x / p_camera.z;
        let y = p_camera.y / p_camera.z;
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        match *self {
            CameraModel::SimplePinhole { .. } | CameraModel::Pinhole { .. } => {
                (fx * x + cx, fy * y + cy)
            }
            CameraModel::SimpleRadial { k, .. } => {
                let r2 = x * x + y * y;
                let d = 1.0 + k * r2;
                (fx * (x * d) + cx, fy * (y * d) + cy)
            }
            CameraModel::Radial { k1, k2, .. } => {
                let r2 = x * x + y * y;
                let d = 1.0 + k1 * r2 + k2 * r2 * r2;
                (fx * (x * d) + cx, fy * (y * d) + cy)
            }
            CameraModel::OpenCV { k1, k2, p1, p2, .. } => {
                let r2 = x * x + y * y;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
                (fx * xd + cx, fy * yd + cy)
            }
            CameraModel::OpenCVFisheye { k1, k2, k3, k4, .. } => {
                let r = (x * x + y * y).sqrt().max(1e-12);
                let theta = r.atan();
                let theta2 = theta * theta;
                let theta_d = theta
                    * (1.0
                        + k1 * theta2
                        + k2 * theta2.powi(2)
                        + k3 * theta2.powi(3)
                        + k4 * theta2.powi(4));
                let scale = theta_d / r;
                (fx * (x * scale) + cx, fy * (y * scale) + cy)
            }
        }
    }
}
