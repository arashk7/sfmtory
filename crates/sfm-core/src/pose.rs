//! Rigid camera pose, stored in COLMAP's world-to-camera convention:
//! `X_cam = R * X_world + t`, with `R` a Hamilton quaternion (qw, qx, qy, qz).
//!
//! COLMAP/OpenCV camera axes: +X right, +Y down, +Z forward (into the scene).
//! NeRF/OpenGL/Blender camera axes: +X right, +Y up, +Z backward (out of the
//! scene). Converting between the two ecosystems means both inverting the pose
//! (world-to-camera -> camera-to-world) *and* flipping the Y/Z camera axes -
//! getting only one of those right is the most common bug in COLMAP<->NeRF
//! converters, so [`Pose::to_nerf_c2w`] / [`Pose::from_nerf_c2w`] do both.

use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// World-to-camera rotation.
    pub rotation: UnitQuaternion<f64>,
    /// World-to-camera translation: `t = -R * camera_center`.
    pub translation: Vector3<f64>,
}

impl Pose {
    pub fn identity() -> Self {
        Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn from_rotation_translation(
        rotation: UnitQuaternion<f64>,
        translation: Vector3<f64>,
    ) -> Self {
        Pose {
            rotation,
            translation,
        }
    }

    /// Camera center in world coordinates: `C = -R^T * t`.
    pub fn camera_center(&self) -> Vector3<f64> {
        -(self.rotation.inverse() * self.translation)
    }

    /// Transform a world-space point into camera space.
    pub fn transform_point(&self, p_world: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * p_world + self.translation
    }

    /// World-to-camera as a 4x4 homogeneous matrix.
    pub fn to_matrix(&self) -> Matrix4<f64> {
        let r: Rotation3<f64> = self.rotation.to_rotation_matrix();
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn from_matrix(m: &Matrix4<f64>) -> Self {
        let r = Matrix3::from(m.fixed_view::<3, 3>(0, 0));
        let t = Vector3::from(m.fixed_view::<3, 1>(0, 3));
        Pose {
            rotation: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r)),
            translation: t,
        }
    }

    /// COLMAP-order Hamilton quaternion (qw, qx, qy, qz).
    pub fn quaternion_wxyz(&self) -> [f64; 4] {
        let q = self.rotation.quaternion();
        [q.w, q.i, q.j, q.k]
    }

    pub fn from_quaternion_wxyz_translation(q: [f64; 4], t: Vector3<f64>) -> Self {
        let quat = nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]);
        Pose {
            rotation: UnitQuaternion::from_quaternion(quat),
            translation: t,
        }
    }

    /// Convert to a NeRF/`transforms.json`-style camera-to-world 4x4 matrix:
    /// inverts the pose (world-to-camera -> camera-to-world) and flips the Y
    /// and Z camera axes (OpenCV convention -> OpenGL/Blender convention).
    pub fn to_nerf_c2w(&self) -> Matrix4<f64> {
        let w2c = self.to_matrix();
        let c2w = w2c
            .try_inverse()
            .expect("valid rotation is always invertible");
        // Flip Y and Z columns of the rotation part (axis convention change);
        // translation (camera center) is unaffected by an axis flip of the frame.
        let mut out = c2w;
        for row in 0..3 {
            out[(row, 1)] = -c2w[(row, 1)];
            out[(row, 2)] = -c2w[(row, 2)];
        }
        out
    }

    /// Inverse of [`Pose::to_nerf_c2w`]: build a world-to-camera `Pose` from a
    /// NeRF-style camera-to-world matrix.
    pub fn from_nerf_c2w(c2w: &Matrix4<f64>) -> Self {
        let mut flipped = *c2w;
        for row in 0..3 {
            flipped[(row, 1)] = -c2w[(row, 1)];
            flipped[(row, 2)] = -c2w[(row, 2)];
        }
        let w2c = flipped
            .try_inverse()
            .expect("valid rotation is always invertible");
        Pose::from_matrix(&w2c)
    }
}
