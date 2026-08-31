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
    /// params: f, cx, cy, k1, k2, k3
    ///
    /// `Radial` with a third radial term. The extra coefficient matters on
    /// wide lenses, where distortion stops being well described by a
    /// quadratic in `r^2` before the projection stops being rectilinear -
    /// leaving a gap between `RADIAL` and `OPENCV_FISHEYE` that neither
    /// covers well.
    Radial3 {
        f: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
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
            CameraModel::Radial3 { .. } => "RADIAL3",
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
            CameraModel::Radial3 { .. } => 4,
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
            "RADIAL3" => CameraModel::Radial3 {
                f: params[0],
                cx: params[1],
                cy: params[2],
                k1: params[3],
                k2: params[4],
                k3: params[5],
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
            CameraModel::Radial3 {
                f,
                cx,
                cy,
                k1,
                k2,
                k3,
            } => vec![f, cx, cy, k1, k2, k3],
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
            CameraModel::Radial3 { f, .. } => (f, f),
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
            | CameraModel::Radial3 { cx, cy, .. }
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
            CameraModel::Radial3 { k1, k2, k3, .. } => [k1, k2, 0.0, 0.0, k3, 0.0],
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
            CameraModel::Radial3 { k1, k2, k3, .. } => {
                let r2 = x * x + y * y;
                let d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
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

#[cfg(test)]
mod radial3_tests {
    use super::*;

    #[test]
    fn radial3_round_trips_through_its_name_and_params() {
        let cam = CameraModel::Radial3 {
            f: 1234.5,
            cx: 640.0,
            cy: 512.0,
            k1: -0.21,
            k2: 0.04,
            k3: -0.006,
        };
        assert_eq!(cam.name(), "RADIAL3");
        let params = cam.params();
        assert_eq!(params.len(), 6);
        assert_eq!(
            CameraModel::from_name_and_params("RADIAL3", &params),
            Some(cam)
        );
        // The third coefficient must survive the OpenCV-ordered view, which
        // puts k3 in slot 4 (after the two tangential terms this model lacks).
        assert_eq!(
            cam.opencv_distortion(),
            [-0.21, 0.04, 0.0, 0.0, -0.006, 0.0]
        );
    }

    /// With `k3 = 0` it must reduce exactly to `RADIAL`, or the extra term is
    /// changing the model rather than extending it.
    #[test]
    fn radial3_with_zero_k3_matches_radial() {
        let r = CameraModel::Radial {
            f: 900.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.3,
            k2: 0.09,
        };
        let r3 = CameraModel::Radial3 {
            f: 900.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.3,
            k2: 0.09,
            k3: 0.0,
        };
        for (x, y, z) in [(0.1, 0.2, 3.0), (-0.4, 0.35, 2.0), (0.0, 0.0, 1.5)] {
            let p = nalgebra::Vector3::new(x, y, z);
            let (ur, vr) = r.project(&p);
            let (u3, v3) = r3.project(&p);
            assert!((ur - u3).abs() < 1e-12 && (vr - v3).abs() < 1e-12);
        }
    }

    /// The third term has to actually bend the projection further out, or it
    /// buys nothing over `RADIAL`.
    #[test]
    fn the_third_term_acts_where_the_second_cannot() {
        let base = CameraModel::Radial3 {
            f: 900.0,
            cx: 320.0,
            cy: 240.0,
            k1: -0.3,
            k2: 0.09,
            k3: 0.0,
        };
        let CameraModel::Radial3 { k1, k2, .. } = base else {
            unreachable!()
        };
        let bent = CameraModel::Radial3 {
            f: 900.0,
            cx: 320.0,
            cy: 240.0,
            k1,
            k2,
            k3: -0.05,
        };
        // Near the axis the r^6 term is negligible; far out it is not.
        let near = nalgebra::Vector3::new(0.02, 0.0, 1.0);
        let far = nalgebra::Vector3::new(0.9, 0.0, 1.0);
        let d_near = (base.project(&near).0 - bent.project(&near).0).abs();
        let d_far = (base.project(&far).0 - bent.project(&far).0).abs();
        assert!(d_near < 1e-6, "near-axis shift {d_near}");
        assert!(d_far > 1.0, "far-field shift {d_far} should be pixels");
    }
}
