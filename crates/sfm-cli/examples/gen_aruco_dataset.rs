//! Renders a synthetic captures x cameras ArUco dataset for exercising the
//! `feature --merge-multicaps` / `--find-params` paths end to end.
//!
//! Usage: gen_aruco_dataset <out_dir> <captures> <cameras> [dim_factor]

use image::{GrayImage, Luma};

const GRID: usize = 6;
const DATA_BITS: usize = 4;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = std::path::PathBuf::from(&a[1]);
    let captures: u32 = a[2].parse().unwrap();
    let cameras: u32 = a[3].parse().unwrap();
    let dim: f32 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);

    let dict = sfm_features::aruco::dictionary(50);
    let canvas = 640u32;

    for cap in 0..captures {
        for cam in 0..cameras {
            let dir = out.join(format!("capture_{cap:03}/cam{cam:03}"));
            std::fs::create_dir_all(&dir).unwrap();

            let mut img = GrayImage::from_pixel(canvas, canvas, Luma([245]));
            // Each capture places a different pair of markers at a different
            // spot; each camera sees them from a slightly different offset, so
            // cameras within a capture share markers (matchable) while
            // captures do not (correctly unmatchable).
            for slot in 0..2u32 {
                let marker_id = (cap * 2 + slot) as usize % 50;
                let code = dict[marker_id];
                let cell = 18u32;
                let ox = 60 + slot * 260 + cam * 12;
                let oy = 90 + (cap % 2) * 200 + cam * 9;
                for gy in 0..GRID {
                    for gx in 0..GRID {
                        let border = gy == 0 || gx == 0 || gy == GRID - 1 || gx == GRID - 1;
                        let on = if border {
                            true
                        } else {
                            (code >> ((gy - 1) * DATA_BITS + (gx - 1))) & 1 == 1
                        };
                        let v = if on { 20u8 } else { 235u8 };
                        for py in 0..cell {
                            for px in 0..cell {
                                let x = ox + gx as u32 * cell + px;
                                let y = oy + gy as u32 * cell + py;
                                if x < canvas && y < canvas {
                                    img.put_pixel(x, y, Luma([v]));
                                }
                            }
                        }
                    }
                }
            }
            // Optional contrast compression toward mid-grey, to give
            // --find-params something real to correct for. Compressing the
            // range (rather than just dimming) is what actually defeats
            // adaptive thresholding: the local mean still tracks the image,
            // but no pixel is far enough below it to read as ink.
            if dim != 1.0 {
                for p in img.pixels_mut() {
                    let v = 128.0 + (p[0] as f32 - 128.0) * dim;
                    p[0] = v.clamp(0.0, 255.0) as u8;
                }
            }
            img.save(dir.join("image.png")).unwrap();
        }
    }
    println!("wrote {captures} captures x {cameras} cameras to {}", out.display());
}
