//! The reconstruction as the viewer needs it, plus the software renderer that
//! draws it.
//!
//! Rendering is done on the CPU into an image that egui then displays as a
//! texture, rather than through a GPU pipeline. A sparse reconstruction is a
//! few tens of thousands of points and a handful of camera frusta - well
//! inside what a straightforward rasteriser handles at interactive rates - and
//! doing it this way keeps the viewer free of shaders, GL state and a second
//! rendering path to keep working.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};
use sfm_core::{Camera, Reconstruction};

/// One registered image: its pose, and enough context to describe it.
pub struct ImageView {
    pub id: u32,
    pub name: String,
    pub camera_id: u32,
    /// Camera centre in world coordinates.
    pub center: Vector3<f64>,
    /// Camera-to-world rotation (the transpose of the stored world-to-camera).
    pub r_cw: Matrix3<f64>,
    pub num_observations: usize,
    pub quaternion: [f64; 4],
    pub translation: [f64; 3],
}

pub struct Scene {
    pub points: Vec<(Vector3<f32>, [u8; 3])>,
    pub images: Vec<ImageView>,
    pub cameras: BTreeMap<u32, Camera>,
    /// Centroid of the reconstruction, used to frame the initial view.
    pub centroid: Vector3<f64>,
    /// Rough radius, so the default camera distance suits any scene scale.
    pub extent: f64,
}

impl Scene {
    pub fn from_reconstruction(recon: &Reconstruction) -> Self {
        let points: Vec<(Vector3<f32>, [u8; 3])> = recon
            .points3d
            .values()
            .map(|p| (Vector3::new(p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32), p.color))
            .collect();

        let mut images: Vec<ImageView> = recon
            .images
            .values()
            .map(|im| {
                let r = im.pose.rotation.to_rotation_matrix().into_inner();
                let q = im.pose.rotation.quaternion();
                ImageView {
                    id: im.id,
                    name: im.name.clone(),
                    camera_id: im.camera_id,
                    center: im.pose.camera_center(),
                    r_cw: r.transpose(),
                    num_observations: im.point3d_ids.iter().filter(|p| p.is_some()).count(),
                    quaternion: [q.w, q.i, q.j, q.k],
                    translation: [
                        im.pose.translation.x,
                        im.pose.translation.y,
                        im.pose.translation.z,
                    ],
                }
            })
            .collect();
        images.sort_by(|a, b| a.name.cmp(&b.name));

        // Frame on the points when there are any, otherwise on the cameras -
        // a model that failed to triangulate anything should still show its
        // poses rather than an empty view.
        let anchors: Vec<Vector3<f64>> = if !points.is_empty() {
            points.iter().map(|(p, _)| p.cast::<f64>()).collect()
        } else {
            images.iter().map(|i| i.center).collect()
        };
        let centroid = if anchors.is_empty() {
            Vector3::zeros()
        } else {
            anchors.iter().sum::<Vector3<f64>>() / anchors.len() as f64
        };
        // A robust radius: the median distance rather than the maximum, so one
        // stray point cannot push the whole scene into the distance.
        let mut d: Vec<f64> = anchors.iter().map(|p| (p - centroid).norm()).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let extent = if d.is_empty() {
            1.0
        } else {
            (d[d.len() * 9 / 10].max(1e-6)) * 1.5
        };

        Scene {
            points,
            images,
            cameras: recon.cameras.clone(),
            centroid,
            extent,
        }
    }
}

/// Orbit camera: yaw/pitch around a target at a given distance.
#[derive(Clone, Copy)]
pub struct OrbitCamera {
    pub target: Vector3<f64>,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub fov_y: f64,
}

impl OrbitCamera {
    pub fn framing(scene: &Scene) -> Self {
        OrbitCamera {
            target: scene.centroid,
            distance: scene.extent * 2.5,
            yaw: 0.6,
            pitch: 0.35,
            fov_y: 50f64.to_radians(),
        }
    }

    pub fn eye(&self) -> Vector3<f64> {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        self.target + Vector3::new(cp * sy, sp, cp * cy) * self.distance
    }

    /// World-to-view rotation, rows being the view axes.
    pub fn view_rotation(&self) -> Matrix3<f64> {
        let fwd = (self.target - self.eye()).normalize();
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        // Looking straight down the up axis leaves `right` undefined; fall back
        // to another reference so the view does not collapse at the poles.
        let up_ref = if fwd.y.abs() > 0.999 {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            world_up
        };
        let right = fwd.cross(&up_ref).normalize();
        let up = right.cross(&fwd);
        Matrix3::from_rows(&[right.transpose(), (-up).transpose(), fwd.transpose()])
    }
}

/// A screen-space line segment.
pub type Edge = ([f32; 2], [f32; 2]);

/// What was drawn, so the UI can hit-test against it without re-projecting.
pub struct Projected {
    /// Screen position of each image's camera centre, when in front of the eye.
    pub image_screen: Vec<(usize, [f32; 2])>,
    /// Frustum edges in screen space, per image.
    pub frusta: Vec<(usize, Vec<Edge>)>,
}

pub struct RenderOptions {
    pub show_points: bool,
    pub show_cameras: bool,
    pub point_size: i32,
    pub bg: [u8; 3],
}

/// Rasterises the scene into an RGBA buffer and reports what landed where.
pub fn render(
    scene: &Scene,
    cam: &OrbitCamera,
    width: usize,
    height: usize,
    opts: &RenderOptions,
) -> (Vec<u8>, Projected) {
    let mut rgba = vec![0u8; width * height * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[0] = opts.bg[0];
        px[1] = opts.bg[1];
        px[2] = opts.bg[2];
        px[3] = 255;
    }
    let mut depth = vec![f32::INFINITY; width * height];

    let eye = cam.eye();
    let r = cam.view_rotation();
    let f = (height as f64 * 0.5) / (cam.fov_y * 0.5).tan();
    let (cx, cy) = (width as f64 * 0.5, height as f64 * 0.5);

    let project = |p: Vector3<f64>| -> Option<([f32; 2], f32)> {
        let v = r * (p - eye);
        if v.z <= 1e-6 {
            return None;
        }
        let x = f * v.x / v.z + cx;
        let y = f * v.y / v.z + cy;
        Some(([x as f32, y as f32], v.z as f32))
    };

    if opts.show_points {
        let rad = opts.point_size.max(1) - 1;
        for (p, color) in &scene.points {
            let Some((s, z)) = project(p.cast::<f64>()) else {
                continue;
            };
            let (sx, sy) = (s[0].round() as i64, s[1].round() as i64);
            for dy in -(rad as i64)..=(rad as i64) {
                for dx in -(rad as i64)..=(rad as i64) {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                        continue;
                    }
                    let idx = y as usize * width + x as usize;
                    // Depth test so nearer points win, which is what makes a
                    // dense cloud read as a surface rather than a haze.
                    if z < depth[idx] {
                        depth[idx] = z;
                        let o = idx * 4;
                        rgba[o] = color[0];
                        rgba[o + 1] = color[1];
                        rgba[o + 2] = color[2];
                    }
                }
            }
        }
    }

    // Camera frusta are returned as line lists for egui to stroke, rather than
    // rasterised here: a handful of crisp anti-aliased lines on top of the
    // point texture looks far better than stair-stepped ones baked into it.
    let mut image_screen = Vec::new();
    let mut frusta = Vec::new();
    if opts.show_cameras {
        let size = scene.extent * 0.06;
        for (i, im) in scene.images.iter().enumerate() {
            let Some((s, _)) = project(im.center) else {
                continue;
            };
            image_screen.push((i, s));
            // Frustum corners one unit forward in the image's own frame.
            let corners_cam = [
                Vector3::new(-0.8, -0.6, 1.4),
                Vector3::new(0.8, -0.6, 1.4),
                Vector3::new(0.8, 0.6, 1.4),
                Vector3::new(-0.8, 0.6, 1.4),
            ];
            let mut pts = Vec::with_capacity(4);
            let mut ok = true;
            for c in corners_cam {
                let world = im.center + im.r_cw * (c * size);
                match project(world) {
                    Some((p, _)) => pts.push(p),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let mut edges = Vec::with_capacity(8);
            for k in 0..4 {
                edges.push((pts[k], pts[(k + 1) % 4]));
                edges.push((s, pts[k]));
            }
            frusta.push((i, edges));
        }
    }

    (
        rgba,
        Projected {
            image_screen,
            frusta,
        },
    )
}
