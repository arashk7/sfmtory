//! The match graph: images as nodes, geometrically-verified pairs as edges.
//!
//! Worth a view of its own because the failure it exposes is invisible in the
//! stage logs. On a 200-image fiducial capture, 49 images went unregistered
//! not because their features or matches were bad but because they formed
//! their own connected component, with no verified pair bridging to the
//! component the reconstruction seeded from. The logs report per-pair inlier
//! counts, all healthy; the *shape* of the graph is the thing that is wrong,
//! and a picture of it says so at a glance.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;

use crate::db::Database;

pub struct Node {
    pub image_id: u32,
    pub name: String,
    /// Index into `MatchGraph::components`.
    pub component: usize,
    /// Whether this image made it into the reconstruction.
    pub registered: bool,
    /// Layout position, in a roughly unit-square coordinate space.
    pub pos: [f32; 2],
    pub degree: usize,
}

pub struct EdgeLink {
    pub a: usize,
    pub b: usize,
    pub inliers: usize,
}

pub struct MatchGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<EdgeLink>,
    /// Node indices per connected component, largest first.
    pub components: Vec<Vec<usize>>,
    pub max_inliers: usize,
}

impl MatchGraph {
    /// Reads the graph straight from the project database, so it is available
    /// after `match` and before `map` has ever run - which is exactly when a
    /// disconnected graph is worth knowing about.
    pub fn load(db_path: &Path, registered: &std::collections::BTreeSet<u32>) -> Result<Self> {
        let db = Database::open(db_path)?;
        let images = db.list_images()?;
        let index_of: BTreeMap<u32, usize> = images
            .iter()
            .enumerate()
            .map(|(i, (id, ..))| (*id, i))
            .collect();

        let mut edges = Vec::new();
        let mut degree = vec![0usize; images.len()];
        for (id1, id2) in db.list_geometry_pairs()? {
            let (Some(&a), Some(&b)) = (index_of.get(&id1), index_of.get(&id2)) else {
                continue;
            };
            // A pair row exists only once it has passed geometric
            // verification, so its inlier count is the edge weight directly.
            let inliers = db
                .load_geometry(id1, id2)
                .map(|g| g.inlier_matches.len())
                .unwrap_or(0);
            degree[a] += 1;
            degree[b] += 1;
            edges.push(EdgeLink { a, b, inliers });
        }

        let component_of = connected_components(images.len(), &edges);
        let num_components = component_of.iter().copied().max().map_or(0, |m| m + 1);
        let mut components: Vec<Vec<usize>> = vec![Vec::new(); num_components];
        for (i, &c) in component_of.iter().enumerate() {
            components[c].push(i);
        }
        // Largest first, so component 0 is the one the reconstruction most
        // likely seeded from and the colour legend reads in a useful order.
        components.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let mut rank = vec![0usize; images.len()];
        for (r, comp) in components.iter().enumerate() {
            for &n in comp {
                rank[n] = r;
            }
        }

        let mut nodes: Vec<Node> = images
            .iter()
            .enumerate()
            .map(|(i, (id, _, name, ..))| Node {
                image_id: *id,
                name: name.clone(),
                component: rank[i],
                registered: registered.contains(id),
                pos: [0.0, 0.0],
                degree: degree[i],
            })
            .collect();

        layout(&mut nodes, &edges, &components);
        let max_inliers = edges.iter().map(|e| e.inliers).max().unwrap_or(1).max(1);

        Ok(MatchGraph {
            nodes,
            edges,
            components,
            max_inliers,
        })
    }

    /// Images with no verified pair at all - the extreme case of the same
    /// problem, and the one most worth naming explicitly.
    pub fn isolated(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].degree == 0)
            .collect()
    }
}

fn connected_components(n: usize, edges: &[EdgeLink]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for e in edges {
        let (ra, rb) = (find(&mut parent, e.a), find(&mut parent, e.b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    // Relabel roots to a dense 0..k range, in first-seen order.
    let mut label = BTreeMap::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = find(&mut parent, i);
        let next = label.len();
        out.push(*label.entry(r).or_insert(next));
    }
    out
}

/// Fruchterman-Reingold, seeded deterministically and run for a fixed number
/// of iterations.
///
/// Deterministic on purpose: a layout that reshuffles every time the model is
/// reloaded makes it impossible to tell whether the *graph* changed or only
/// the picture of it. Each component is laid out in its own disc, so
/// components stay visually separate however the force simulation settles -
/// separation is the whole point of the view, and leaving it to repulsion
/// alone would let a large component's forces swamp a small one.
fn layout(nodes: &mut [Node], edges: &[EdgeLink], components: &[Vec<usize>]) {
    if nodes.is_empty() {
        return;
    }
    // Place component discs around a circle, sized by member count.
    let k = components.len().max(1);
    let mut centres = Vec::with_capacity(k);
    let mut radii = Vec::with_capacity(k);
    for (i, comp) in components.iter().enumerate() {
        let frac = (comp.len() as f32 / nodes.len() as f32).sqrt();
        if k == 1 {
            centres.push([0.0f32, 0.0f32]);
        } else {
            let a = i as f32 / k as f32 * std::f32::consts::TAU;
            centres.push([a.cos() * 0.7, a.sin() * 0.7]);
        }
        radii.push((frac * 0.45).max(0.03));
    }

    // Deterministic pseudo-random start: a golden-angle spiral inside each
    // component's disc, which is well-spread without needing an RNG.
    for (ci, comp) in components.iter().enumerate() {
        for (j, &n) in comp.iter().enumerate() {
            let t = (j as f32 + 0.5) / comp.len() as f32;
            let a = j as f32 * 2.399_963_2; // golden angle, radians
            nodes[n].pos = [
                centres[ci][0] + t.sqrt() * radii[ci] * a.cos(),
                centres[ci][1] + t.sqrt() * radii[ci] * a.sin(),
            ];
        }
    }

    let n = nodes.len();
    // O(n^2) repulsion is fine at the scale this view is for: the largest real
    // capture measured here is 200 images, so 40k pair terms per iteration.
    if n > 600 {
        return;
    }
    let area = 1.0f32;
    let ideal = (area / n as f32).sqrt();
    let mut disp = vec![[0.0f32; 2]; n];
    const ITERATIONS: usize = 120;
    for it in 0..ITERATIONS {
        for d in disp.iter_mut() {
            *d = [0.0, 0.0];
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = nodes[i].pos[0] - nodes[j].pos[0];
                let dy = nodes[i].pos[1] - nodes[j].pos[1];
                let d2 = (dx * dx + dy * dy).max(1e-6);
                let f = ideal * ideal / d2;
                disp[i][0] += dx * f;
                disp[i][1] += dy * f;
                disp[j][0] -= dx * f;
                disp[j][1] -= dy * f;
            }
        }
        for e in edges {
            let dx = nodes[e.a].pos[0] - nodes[e.b].pos[0];
            let dy = nodes[e.a].pos[1] - nodes[e.b].pos[1];
            let d = (dx * dx + dy * dy).sqrt().max(1e-4);
            let f = d / ideal;
            disp[e.a][0] -= dx / d * f * ideal;
            disp[e.a][1] -= dy / d * f * ideal;
            disp[e.b][0] += dx / d * f * ideal;
            disp[e.b][1] += dy / d * f * ideal;
        }
        // Cooling schedule, so early iterations move freely and later ones
        // only settle.
        let temp = 0.1 * (1.0 - it as f32 / ITERATIONS as f32);
        for i in 0..n {
            let d = (disp[i][0] * disp[i][0] + disp[i][1] * disp[i][1])
                .sqrt()
                .max(1e-6);
            let step = d.min(temp) / d;
            nodes[i].pos[0] += disp[i][0] * step;
            nodes[i].pos[1] += disp[i][1] * step;
            // Pull each node gently back toward its own component's disc, so
            // the components cannot drift through one another.
            let c = centres[nodes[i].component];
            nodes[i].pos[0] += (c[0] - nodes[i].pos[0]) * 0.01;
            nodes[i].pos[1] += (c[1] - nodes[i].pos[1]) * 0.01;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(a: usize, b: usize) -> EdgeLink {
        EdgeLink { a, b, inliers: 10 }
    }

    #[test]
    fn components_split_a_disconnected_graph() {
        // 0-1-2 and 3-4, plus an isolated 5.
        let edges = [e(0, 1), e(1, 2), e(3, 4)];
        let c = connected_components(6, &edges);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[1], c[2]);
        assert_eq!(c[3], c[4]);
        assert_ne!(c[0], c[3]);
        assert_ne!(c[0], c[5]);
        assert_ne!(c[3], c[5]);
        assert_eq!(c.iter().copied().max().unwrap() + 1, 3);
    }

    #[test]
    fn a_fully_connected_graph_is_one_component() {
        let edges = [e(0, 1), e(1, 2), e(2, 3), e(3, 0)];
        let c = connected_components(4, &edges);
        assert!(c.iter().all(|&x| x == c[0]));
    }

    #[test]
    fn layout_is_deterministic_and_keeps_components_apart() {
        let edges = vec![e(0, 1), e(1, 2), e(3, 4)];
        let components = vec![vec![0, 1, 2], vec![3, 4]];
        let make = || {
            (0..5)
                .map(|i| Node {
                    image_id: i as u32,
                    name: format!("{i}"),
                    component: if i < 3 { 0 } else { 1 },
                    registered: true,
                    pos: [0.0, 0.0],
                    degree: 2,
                })
                .collect::<Vec<_>>()
        };
        let mut a = make();
        let mut b = make();
        layout(&mut a, &edges, &components);
        layout(&mut b, &edges, &components);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.pos, y.pos, "layout must not vary between runs");
        }
        // Every member of one component ends up nearer its own centroid than
        // the other component's.
        let centroid = |ns: &[Node], comp: &[usize]| {
            let (mut x, mut y) = (0.0f32, 0.0f32);
            for &i in comp {
                x += ns[i].pos[0];
                y += ns[i].pos[1];
            }
            [x / comp.len() as f32, y / comp.len() as f32]
        };
        let c0 = centroid(&a, &components[0]);
        let c1 = centroid(&a, &components[1]);
        let d = |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        for &i in &components[0] {
            assert!(d(a[i].pos, c0) < d(a[i].pos, c1), "node {i} drifted");
        }
    }
}
