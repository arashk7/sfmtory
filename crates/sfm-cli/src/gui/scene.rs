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
    /// Point ids, parallel to `points`, so a pick or a residual lookup can go
    /// from a rendered pixel back to the model's own `Point3D`.
    pub point_ids: Vec<u64>,
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
            .map(|p| {
                (
                    Vector3::new(p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32),
                    p.color,
                )
            })
            .collect();
        let point_ids: Vec<u64> = recon.points3d.values().map(|p| p.id).collect();

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
            point_ids,
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
    /// Index into `Scene::points` for whichever point owns each pixel, or -1.
    ///
    /// Picking reuses the rasteriser's own depth test rather than re-projecting
    /// and searching for the nearest hit in the UI layer: with a dense cloud the
    /// nearest point *on screen* is routinely one occluded behind the surface
    /// being clicked, and only the buffer that already resolved occlusion knows
    /// which one is actually visible. Empty when points are hidden.
    pub point_at: Vec<i32>,
    /// The fitted plane's outline, when one is being shown.
    pub plane_edges: Vec<Edge>,
}

/// A plane to draw as a wireframe, in world coordinates.
pub struct PlaneOverlay {
    pub centroid: Vector3<f64>,
    pub basis: [Vector3<f64>; 2],
    pub half_extent: f64,
}

pub struct RenderOptions<'a> {
    pub show_points: bool,
    pub show_cameras: bool,
    pub point_size: i32,
    pub bg: [u8; 3],
    /// Per-point colour override, parallel to `Scene::points`. `None` draws
    /// each point in its own photometric colour.
    pub point_colors: Option<&'a [[u8; 3]]>,
    /// Per-point visibility, parallel to `Scene::points`. `None` shows all.
    pub point_visible: Option<&'a [bool]>,
    pub plane: Option<&'a PlaneOverlay>,
}

impl Default for RenderOptions<'_> {
    fn default() -> Self {
        RenderOptions {
            show_points: true,
            show_cameras: true,
            point_size: 2,
            bg: [22, 24, 28],
            point_colors: None,
            point_visible: None,
            plane: None,
        }
    }
}

/// Rasterises the scene into an RGBA buffer and reports what landed where.
pub fn render(
    scene: &Scene,
    cam: &OrbitCamera,
    width: usize,
    height: usize,
    opts: &RenderOptions<'_>,
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

    let mut point_at: Vec<i32> = Vec::new();
    if opts.show_points {
        point_at = vec![-1i32; width * height];
        let rad = opts.point_size.max(1) - 1;
        for (i, (p, own_color)) in scene.points.iter().enumerate() {
            if opts.point_visible.is_some_and(|v| !v[i]) {
                continue;
            }
            let color = opts.point_colors.map_or(own_color, |c| &c[i]);
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
                        point_at[idx] = i as i32;
                        let o = idx * 4;
                        rgba[o] = color[0];
                        rgba[o + 1] = color[1];
                        rgba[o + 2] = color[2];
                    }
                }
            }
        }
    }

    // The fitted plane, as a wireframe grid. Drawn as edges for the same
    // reason the frusta are: crisp anti-aliased lines from egui beat
    // stair-stepped ones baked into the texture.
    let mut plane_edges = Vec::new();
    if let Some(pl) = opts.plane {
        const DIVISIONS: usize = 8;
        let step = pl.half_extent * 2.0 / DIVISIONS as f64;
        let corner = pl.centroid - (pl.basis[0] + pl.basis[1]) * pl.half_extent;
        for k in 0..=DIVISIONS {
            let t = k as f64 * step;
            for (along, across) in [(0usize, 1usize), (1, 0)] {
                let a = corner + pl.basis[across] * t;
                let b = a + pl.basis[along] * (pl.half_extent * 2.0);
                if let (Some((pa, _)), Some((pb, _))) = (project(a), project(b)) {
                    plane_edges.push((pa, pb));
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
            point_at,
            plane_edges,
        },
    )
}

/// Index into `Scene::points` of whatever is visible at `(x, y)`.
pub fn pick_point(
    projected: &Projected,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> Option<usize> {
    if projected.point_at.is_empty() {
        return None;
    }
    // Search outward a little: a single point is a few pixels across at most,
    // and demanding an exact hit makes selection feel broken.
    const RADIUS: i64 = 4;
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    let mut best: Option<(i64, usize)> = None;
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let (px, py) = (cx + dx, cy + dy);
            if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                continue;
            }
            let v = projected.point_at[py as usize * width + px as usize];
            if v < 0 {
                continue;
            }
            let d2 = dx * dx + dy * dy;
            if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, v as usize));
            }
        }
    }
    best.map(|(_, i)| i)
}

/// Maps a residual in pixels onto a colour ramp running blue -> green ->
/// yellow -> red over `0..=max`.
///
/// A ramp rather than a two-colour threshold because the question the view has
/// to answer is whether a model is uniformly decent or mostly excellent with a
/// few broken points - `temple_ring` has a 0.283px mean and a 572px maximum,
/// and a binary good/bad colouring at any threshold hides one or the other.
pub fn error_color(residual_px: f64, max: f64) -> [u8; 3] {
    let t = if max > 0.0 {
        (residual_px / max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Piecewise-linear through four stops.
    const STOPS: [[f32; 3]; 4] = [
        [40.0, 110.0, 240.0],
        [40.0, 200.0, 120.0],
        [245.0, 210.0, 60.0],
        [230.0, 60.0, 50.0],
    ];
    let x = t as f32 * (STOPS.len() - 1) as f32;
    let i = (x.floor() as usize).min(STOPS.len() - 2);
    let f = x - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = (STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_ramp_spans_its_stops_and_clamps() {
        assert_eq!(error_color(0.0, 10.0), [40, 110, 240]);
        assert_eq!(error_color(10.0, 10.0), [230, 60, 50]);
        // Beyond the top of the range still clamps to the last stop rather
        // than wrapping.
        assert_eq!(error_color(1e6, 10.0), [230, 60, 50]);
        // A degenerate range (every residual identical) must not divide by zero.
        assert_eq!(error_color(3.0, 0.0), [40, 110, 240]);
        // Every stop is hit exactly at its own position along the range.
        assert_eq!(error_color(10.0 / 3.0, 10.0), [40, 200, 120]);
        assert_eq!(error_color(20.0 / 3.0, 10.0), [245, 210, 60]);
        // Blue falls monotonically from end to end - the ramp's actual
        // ordering invariant. Red is *not* monotone (yellow is redder than the
        // final red), which is why it cannot be the thing asserted on.
        let mut prev = 256i32;
        for k in 0..=40 {
            let c = error_color(k as f64 * 0.25, 10.0);
            assert!(c[2] as i32 <= prev, "blue channel rose at {k}");
            prev = c[2] as i32;
        }
    }
}
