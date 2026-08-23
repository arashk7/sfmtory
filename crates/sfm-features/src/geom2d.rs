//! Minimal 2D computational-geometry helpers used only by the ArUco-style
//! marker detector: convex hull + minimum-area bounding rectangle, which
//! together turn a blob's border pixels into 4 candidate corner points.

/// Andrew's monotone chain convex hull. Input order doesn't matter; returns
/// hull points in counter-clockwise order with no duplicate closing point.
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area bounding rectangle of a convex polygon via rotating
/// calipers. Returns 4 corners in a consistent (clockwise) order, or `None`
/// if the hull is degenerate (fewer than 3 points).
pub fn min_area_rect(hull: &[(f64, f64)]) -> Option<[(f64, f64); 4]> {
    if hull.len() < 3 {
        return None;
    }
    let n = hull.len();
    let mut best_area = f64::MAX;
    let mut best_corners = None;

    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];
        let edge = (p2.0 - p1.0, p2.1 - p1.1);
        let len = (edge.0 * edge.0 + edge.1 * edge.1).sqrt();
        if len < 1e-9 {
            continue;
        }
        let ux = edge.0 / len;
        let uy = edge.1 / len;
        // Project every hull point onto the edge direction (u) and its
        // perpendicular (v) to get an axis-aligned bounding box in that frame.
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &p in hull {
            let dx = p.0 - p1.0;
            let dy = p.1 - p1.1;
            let u = dx * ux + dy * uy;
            let v = dx * (-uy) + dy * ux;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let to_world = |u: f64, v: f64| (p1.0 + u * ux - v * uy, p1.1 + u * uy + v * ux);
            best_corners = Some([
                to_world(min_u, min_v),
                to_world(max_u, min_v),
                to_world(max_u, max_v),
                to_world(min_u, max_v),
            ]);
        }
    }
    best_corners
}

pub fn polygon_area(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % n];
        sum += x1 * y2 - x2 * y1;
    }
    (sum * 0.5).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_square_with_interior_points_is_the_square() {
        let pts = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 5.0),
            (2.0, 3.0),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!((polygon_area(&hull) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn min_area_rect_of_axis_aligned_square_is_itself() {
        let hull = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let rect = min_area_rect(&hull).unwrap();
        let area = polygon_area(&rect);
        assert!((area - 100.0).abs() < 1e-6, "area={area}");
    }

    #[test]
    fn min_area_rect_of_rotated_square_matches_area() {
        let angle: f64 = 0.3;
        let side = 10.0f64;
        let center = (5.0, 5.0);
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let hull: Vec<(f64, f64)> = corners
            .iter()
            .map(|(x, y)| {
                let hx = x * side / 2.0;
                let hy = y * side / 2.0;
                let rx = hx * angle.cos() - hy * angle.sin();
                let ry = hx * angle.sin() + hy * angle.cos();
                (center.0 + rx, center.1 + ry)
            })
            .collect();
        let rect = min_area_rect(&hull).unwrap();
        let area = polygon_area(&rect);
        assert!((area - 100.0).abs() < 1e-3, "area={area}");
    }
}
