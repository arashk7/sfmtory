//! Renders a geometrically valid 3D ArUco scene: markers fixed to the faces of
//! a cube, photographed from many viewpoints on a surrounding sphere.
//!
//! Unlike a flat test pattern, this produces real perspective and real depth
//! variation, so the reconstruction it feeds is a genuine 3D problem rather
//! than a degenerate planar one. Each marker is rasterised through the exact
//! homography induced by projecting its four 3D corners, so the actual
//! detector runs on actual imagery.
//!
//! Usage: gen_aruco_scene3d <out_images_dir> <num_views> [focal] [width] [height]

use image::{GrayImage, Luma};
use nalgebra::{Matrix3, Vector3};

const GRID: usize = 6;
const DATA_BITS: usize = 4;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = std::path::PathBuf::from(&a[1]);
    let views: usize = a[2].parse().unwrap();
    let focal: f64 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(600.0);
    let w: u32 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(640);
    let h: u32 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(480);
    std::fs::create_dir_all(&out).unwrap();

    let dict_size: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(150);
    let dict = sfm_features::aruco::dictionary(dict_size);

    // Cube faces: origin corner plus the two in-plane edge vectors, and the
    // outward normal. Markers are laid out in a grid on each face.
    let half = 1.0f64;
    let faces: [(Vector3<f64>, Vector3<f64>, Vector3<f64>); 6] = [
        (
            Vector3::new(-half, -half, half),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ), // +z
        (
            Vector3::new(half, -half, -half),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ), // -z
        (
            Vector3::new(half, -half, half),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ), // +x
        (
            Vector3::new(-half, -half, -half),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ), // -x
        (
            Vector3::new(-half, half, half),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ), // +y
        (
            Vector3::new(-half, -half, -half),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ), // -y
    ];

    // (marker_id, 4 corners in 3D, outward normal). Corner order matches the
    // detector's convention: clockwise starting at the marker's top-left when
    // viewed head-on from outside.
    let mut markers: Vec<(usize, [Vector3<f64>; 4], Vector3<f64>)> = Vec::new();
    let per_side = 5usize;
    let msize = 0.30f64;
    let gap = (2.0 * half - per_side as f64 * msize) / (per_side as f64 + 1.0);
    let mut next_id = 0usize;
    for (origin, du, dv) in faces.iter() {
        let normal = du.cross(dv).normalize();
        for iy in 0..per_side {
            for ix in 0..per_side {
                if next_id >= dict_size {
                    continue;
                }
                let u0 = gap + ix as f64 * (msize + gap);
                let v0 = gap + iy as f64 * (msize + gap);
                // Stand each marker off its face by a varying amount. Markers
                // flush against a face are coplanar, and any view pair whose
                // shared markers all lie on one face then gives a degenerate
                // essential matrix - a relative pose with no usable
                // translation. Real fiducial setups are rarely perfectly
                // coplanar either; this keeps the scene honestly 3D.
                let lift = 0.05 + 0.45 * (((next_id * 7) % 5) as f64 / 4.0);
                let origin = origin + normal * lift;
                let p = |uu: f64, vv: f64| origin + du * uu + dv * vv;
                markers.push((
                    next_id,
                    [
                        p(u0, v0 + msize),
                        p(u0 + msize, v0 + msize),
                        p(u0 + msize, v0),
                        p(u0, v0),
                    ],
                    normal,
                ));
                next_id += 1;
            }
        }
    }

    let radius = 4.2f64;
    let mut rendered = 0usize;
    for v in 0..views {
        // Spiral over the sphere so consecutive views overlap heavily (strong
        // covisibility) while the set as a whole covers all sides.
        let t = v as f64 / views as f64;
        let phi = (1.0 - 2.0 * t).acos();
        let theta = t * std::f64::consts::PI * 2.0 * 5.0;
        let eye = Vector3::new(
            radius * phi.sin() * theta.cos(),
            radius * phi.cos() * 0.6,
            radius * phi.sin() * theta.sin(),
        );
        // Look-at the origin; world-to-camera rotation rows are the camera axes.
        let fwd = (-eye).normalize();
        let world_up = if fwd.y.abs() > 0.95 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let right = fwd.cross(&world_up).normalize();
        let up = right.cross(&fwd);
        let r = Matrix3::from_rows(&[right.transpose(), (-up).transpose(), fwd.transpose()]);
        let tvec = -r * eye;

        let mut img = GrayImage::from_pixel(w, h, Luma([245]));
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);

        for (mid, corners, normal) in &markers {
            // Back-face cull: only render markers whose face points at us.
            if normal.normalize().dot(&(eye - corners[0]).normalize()) <= 0.30 {
                continue;
            }
            let mut uv = [[0.0f64; 2]; 4];
            let mut ok = true;
            for (k, c) in corners.iter().enumerate() {
                let pc = r * c + tvec;
                if pc.z <= 0.1 {
                    ok = false;
                    break;
                }
                uv[k] = [focal * pc.x / pc.z + cx, focal * pc.y / pc.z + cy];
            }
            if !ok {
                continue;
            }
            // Require the whole marker on-screen with a margin, so partial
            // markers never become half-detections.
            if uv
                .iter()
                .any(|p| p[0] < 4.0 || p[1] < 4.0 || p[0] > w as f64 - 5.0 || p[1] > h as f64 - 5.0)
            {
                continue;
            }
            // Too small to decode reliably - skip rather than emit noise.
            let area = polygon_area(&uv);
            if area < 700.0 {
                continue;
            }

            let hmat = homography_unit_square_to(&uv);
            let inv = match hmat.try_inverse() {
                Some(i) => i,
                None => continue,
            };
            let code = dict[*mid];
            let (minx, maxx, miny, maxy) = bounds(&uv, w, h);
            for py in miny..=maxy {
                for px in minx..=maxx {
                    let q = inv * Vector3::new(px as f64 + 0.5, py as f64 + 0.5, 1.0);
                    if q.z.abs() < 1e-12 {
                        continue;
                    }
                    let (su, sv) = (q.x / q.z, q.y / q.z);
                    if !(0.0..1.0).contains(&su) || !(0.0..1.0).contains(&sv) {
                        continue;
                    }
                    let gx = (su * GRID as f64) as usize;
                    let gy = (sv * GRID as f64) as usize;
                    let border = gx == 0 || gy == 0 || gx == GRID - 1 || gy == GRID - 1;
                    let on = if border {
                        true
                    } else {
                        (code >> ((gy - 1) * DATA_BITS + (gx - 1))) & 1 == 1
                    };
                    img.put_pixel(px, py, Luma([if on { 20 } else { 235 }]));
                }
            }
            rendered += 1;
        }
        img.save(out.join(format!("view_{v:04}.png"))).unwrap();
    }
    println!(
        "wrote {views} views of {} markers to {} ({rendered} marker renders, ~{:.1} markers/view, focal {focal})",
        markers.len(),
        out.display(),
        rendered as f64 / views as f64
    );
}

fn polygon_area(uv: &[[f64; 2]; 4]) -> f64 {
    let mut a = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        a += uv[i][0] * uv[j][1] - uv[j][0] * uv[i][1];
    }
    a.abs() / 2.0
}

fn bounds(uv: &[[f64; 2]; 4], w: u32, h: u32) -> (u32, u32, u32, u32) {
    let minx = uv
        .iter()
        .map(|p| p[0])
        .fold(f64::MAX, f64::min)
        .floor()
        .max(0.0) as u32;
    let maxx = (uv.iter().map(|p| p[0]).fold(f64::MIN, f64::max).ceil() as i64)
        .clamp(0, w as i64 - 1) as u32;
    let miny = uv
        .iter()
        .map(|p| p[1])
        .fold(f64::MAX, f64::min)
        .floor()
        .max(0.0) as u32;
    let maxy = (uv.iter().map(|p| p[1]).fold(f64::MIN, f64::max).ceil() as i64)
        .clamp(0, h as i64 - 1) as u32;
    (minx, maxx, miny, maxy)
}

/// Homography mapping the unit square (0,0)-(1,1) onto the given quad, solved
/// as the usual 8x8 linear system.
fn homography_unit_square_to(uv: &[[f64; 2]; 4]) -> Matrix3<f64> {
    let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut a = nalgebra::DMatrix::<f64>::zeros(8, 8);
    let mut b = nalgebra::DVector::<f64>::zeros(8);
    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (u, v) = (uv[i][0], uv[i][1]);
        a[(2 * i, 0)] = x;
        a[(2 * i, 1)] = y;
        a[(2 * i, 2)] = 1.0;
        a[(2 * i, 6)] = -x * u;
        a[(2 * i, 7)] = -y * u;
        b[2 * i] = u;
        a[(2 * i + 1, 3)] = x;
        a[(2 * i + 1, 4)] = y;
        a[(2 * i + 1, 5)] = 1.0;
        a[(2 * i + 1, 6)] = -x * v;
        a[(2 * i + 1, 7)] = -y * v;
        b[2 * i + 1] = v;
    }
    let s = a.lu().solve(&b).unwrap();
    Matrix3::new(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1.0)
}
