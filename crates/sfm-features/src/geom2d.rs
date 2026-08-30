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

/// Ramer-Douglas-Peucker simplification of a closed polygon.
///
/// Used to reduce a blob's convex hull to its actual corners. The hull of a
/// square marker has many points along each edge (one per boundary pixel);
/// what the detector needs is the four places where the direction changes.
pub fn approx_poly_closed(points: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    // A closed curve has no natural endpoints, so anchor on the two most
    // distant points and simplify the two chains between them. Anchoring
    // anywhere else can round off a real corner that happens to sit at the
    // arbitrary start of the point list.
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut best = -1.0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = dist2(points[i], points[j]);
            if d > best {
                best = d;
                ia = i;
                ib = j;
            }
        }
    }
    let first: Vec<(f64, f64)> = points[ia..=ib].to_vec();
    let second: Vec<(f64, f64)> = points[ib..]
        .iter()
        .chain(points[..=ia].iter())
        .copied()
        .collect();
    let mut out = rdp(&first, epsilon);
    let mut back = rdp(&second, epsilon);
    // Both chains include the shared anchors; drop the duplicates.
    out.pop();
    back.pop();
    out.append(&mut back);
    out
}

fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

/// Perpendicular distance from `p` to the infinite line through `a` and `b`.
fn line_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return dist2(p, a).sqrt();
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len
}

fn rdp(points: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let (first, last) = (points[0], points[points.len() - 1]);
    let mut worst = (0.0, 0usize);
    for (i, p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = line_distance(*p, first, last);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    if worst.0 <= epsilon {
        return vec![first, last];
    }
    let mut left = rdp(&points[..=worst.1], epsilon);
    let right = rdp(&points[worst.1..], epsilon);
    left.pop();
    left.extend(right);
    left
}

/// Reduces a blob's convex hull to exactly four corners, or reports that it is
/// not a quadrilateral.
///
/// This replaces fitting a minimum-area rectangle, which was wrong twice over.
/// It accepted anything roughly box-shaped - a circle fills 78.5% of its
/// bounding square, comfortably past a 75% area test, which is why round
/// targets were being read as markers. And its corners are those of a *rotated
/// rectangle*, while a marker seen at an angle is a general quadrilateral, so
/// every corner it produced for a tilted marker was systematically displaced -
/// in a pipeline whose entire purpose is to turn those corners into a
/// calibration.
///
/// `epsilon` is swept rather than fixed: the right simplification tolerance
/// depends on how big the marker is in frame, and a single value either rounds
/// the corners off small markers or keeps noise on large ones.
pub fn quad_from_hull(hull: &[(f64, f64)]) -> Option<[(f64, f64); 4]> {
    if hull.len() < 4 {
        return None;
    }
    let perimeter: f64 = (0..hull.len())
        .map(|i| dist2(hull[i], hull[(i + 1) % hull.len()]).sqrt())
        .sum();
    if perimeter < 1e-6 {
        return None;
    }
    for k in 1..=12 {
        let epsilon = perimeter * 0.005 * k as f64;
        let approx = approx_poly_closed(hull, epsilon);
        if approx.len() == 4 {
            let quad = [approx[0], approx[1], approx[2], approx[3]];
            // The four points must also *explain* the hull. Simplification
            // will happily reduce a circle to an inscribed square with four
            // perfect right angles, so shape checks on the quad alone cannot
            // reject one; the giveaway is how much hull area the quad leaves
            // outside itself. An inscribed square covers 2/pi = 64% of its
            // circle, while a real marker's quad covers essentially all of its
            // hull.
            let covered = polygon_area(&quad) / polygon_area(hull).max(1e-9);
            if covered >= 0.85 && is_plausible_quad(&quad) {
                return Some(quad);
            }
            return None;
        }
        if approx.len() < 4 {
            // Past the point of resolving four corners; a coarser tolerance
            // will only merge more.
            break;
        }
    }
    None
}

/// Rejects degenerate quadrilaterals: near-collinear corners, extreme slivers.
fn is_plausible_quad(q: &[(f64, f64); 4]) -> bool {
    let sides: Vec<f64> = (0..4).map(|i| dist2(q[i], q[(i + 1) % 4]).sqrt()).collect();
    let (shortest, longest) = sides
        .iter()
        .fold((f64::MAX, 0.0f64), |(lo, hi), s| (lo.min(*s), hi.max(*s)));
    if shortest < 1e-6 || longest / shortest > 6.0 {
        return false;
    }
    // Interior angles away from straight. A marker viewed at a steep angle
    // still has corners; a blob simplified to four points often does not.
    for i in 0..4 {
        let (a, b, c) = (q[(i + 3) % 4], q[i], q[(i + 1) % 4]);
        let (v1x, v1y) = (a.0 - b.0, a.1 - b.1);
        let (v2x, v2y) = (c.0 - b.0, c.1 - b.1);
        let n1 = (v1x * v1x + v1y * v1y).sqrt();
        let n2 = (v2x * v2x + v2y * v2y).sqrt();
        if n1 < 1e-9 || n2 < 1e-9 {
            return false;
        }
        let cos = ((v1x * v2x + v1y * v2y) / (n1 * n2)).clamp(-1.0, 1.0);
        let deg = cos.acos().to_degrees();
        if !(25.0..=155.0).contains(&deg) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod quad_tests {
    use super::*;

    /// Points along a closed shape, as a blob's hull would give them.
    fn sample_polygon(corners: &[(f64, f64)], per_edge: usize) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for i in 0..corners.len() {
            let a = corners[i];
            let b = corners[(i + 1) % corners.len()];
            for k in 0..per_edge {
                let t = k as f64 / per_edge as f64;
                out.push((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
            }
        }
        out
    }

    #[test]
    fn recovers_the_corners_of_a_square() {
        let corners = [(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)];
        let hull = sample_polygon(&corners, 25);
        let quad = quad_from_hull(&hull).expect("a square is a quad");
        for c in corners {
            assert!(
                quad.iter().any(|q| dist2(*q, c).sqrt() < 2.0),
                "corner {c:?} not recovered in {quad:?}"
            );
        }
    }

    /// The perspective case that a minimum-area rectangle gets wrong: the
    /// recovered corners must be the marker's own, not a bounding box's.
    #[test]
    fn recovers_the_corners_of_a_perspective_quad() {
        let corners = [(20.0, 30.0), (180.0, 10.0), (200.0, 150.0), (40.0, 190.0)];
        let hull = sample_polygon(&corners, 30);
        let quad = quad_from_hull(&hull).expect("still a quad under perspective");
        for c in corners {
            assert!(
                quad.iter().any(|q| dist2(*q, c).sqrt() < 3.0),
                "corner {c:?} not recovered in {quad:?}"
            );
        }
        // A bounding rectangle would have put a corner at the extremes of x
        // and y simultaneously; no real corner of this quad is there.
        assert!(!quad.iter().any(|q| dist2(*q, (200.0, 10.0)).sqrt() < 10.0));
    }

    /// The false positive that motivated all of this: a circle fills 78.5% of
    /// its bounding square and sailed through an area-ratio test.
    #[test]
    fn rejects_a_circle() {
        let circle: Vec<(f64, f64)> = (0..120)
            .map(|i| {
                let a = i as f64 / 120.0 * std::f64::consts::TAU;
                (100.0 + 50.0 * a.cos(), 100.0 + 50.0 * a.sin())
            })
            .collect();
        assert!(quad_from_hull(&circle).is_none(), "a circle is not a quad");
        // An ellipse and a rounded square are the same failure in disguise.
        let ellipse: Vec<(f64, f64)> = (0..120)
            .map(|i| {
                let a = i as f64 / 120.0 * std::f64::consts::TAU;
                (100.0 + 80.0 * a.cos(), 100.0 + 40.0 * a.sin())
            })
            .collect();
        assert!(
            quad_from_hull(&ellipse).is_none(),
            "an ellipse is not a quad"
        );
    }

    #[test]
    fn rejects_triangles_hexagons_and_slivers() {
        let tri = sample_polygon(&[(0.0, 0.0), (100.0, 0.0), (50.0, 90.0)], 30);
        assert!(quad_from_hull(&tri).is_none());

        let hex: Vec<(f64, f64)> = (0..6)
            .map(|i| {
                let a = i as f64 / 6.0 * std::f64::consts::TAU;
                (50.0 * a.cos(), 50.0 * a.sin())
            })
            .collect();
        assert!(quad_from_hull(&sample_polygon(&hex, 20)).is_none());

        // Long thin sliver: four corners, but not a marker.
        let sliver = sample_polygon(&[(0.0, 0.0), (400.0, 0.0), (400.0, 8.0), (0.0, 8.0)], 30);
        assert!(quad_from_hull(&sliver).is_none());
    }

    #[test]
    fn rdp_keeps_endpoints_and_drops_collinear_points() {
        let line: Vec<(f64, f64)> = (0..=10).map(|i| (i as f64, 0.0)).collect();
        assert_eq!(rdp(&line, 0.1), vec![(0.0, 0.0), (10.0, 0.0)]);
    }
}
