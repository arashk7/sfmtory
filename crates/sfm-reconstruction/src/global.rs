//! GLOMAP-style global SfM: view graph -> outlier-pruned rotation averaging
//! (spanning-tree init + chordal-L2/IRLS refinement, cycle-consistency
//! triplet filtering) -> global translation/position averaging (spanning-
//! tree init with an assumed unit baseline per hop + local IRLS
//! consistency refinement) -> track building (union-find over verified
//! matches + angle/error filtering) -> global bundle adjustment, reusing
//! the incremental pipeline's own `run_bundle_adjustment`/
//! `assemble_reconstruction` (see the doc comment on [`run_global`]'s final
//! BA calls for why that reuse is gauge-sound).
//!
//! Known simplifications, documented rather than hidden (mirrors this
//! crate's incremental-pipeline documentation style):
//! - Rotation averaging is **chordal-L2 with IRLS reweighting** (weighted
//!   quaternion averaging, sign-corrected per vote, renormalized), not true
//!   geodesic/Lie-algebra L1 averaging - avoids exp/log-map machinery, same
//!   "fewer lines to get right" tradeoff `sfm-geometry`'s linear 8-point/
//!   6-point solvers already make over minimal solvers.
//! - Translation/position averaging **assumes every view-graph edge has a
//!   roughly comparable baseline length** (seeded as exactly 1.0 per hop,
//!   then only refined for *directional* consistency, never re-derived
//!   from scratch) - see `average_translations`'s doc comment for why a
//!   more "proper" global least-squares solve on direction-only
//!   constraints turned out to be badly conditioned on real data instead.
//!   Not a robust aggregate over per-edge scale the way BATA/LUD-style
//!   solvers recover it - a genuinely wide-baseline pair and a genuinely
//!   narrow one both contribute a unit step.
//! - **No cross-component stitching**: only the largest connected component
//!   (by the post-triplet-filter view graph) is registered; images in other
//!   components are left unregistered, same as an image the incremental
//!   pipeline never manages to register.
//! - **No retriangulation pass**: track building triangulates each point
//!   once from the averaged poses; the final BA's outlier-filter-and-
//!   reoptimize loop (`run_bundle_adjustment`'s `allow_intrinsics` path)
//!   drops bad observations but never recomputes a point's position from
//!   scratch given its surviving track - same documented gap as the
//!   incremental pipeline (see this file's parent module doc comment).
//! - Cycle-consistency triplet filtering can't validate an edge with zero
//!   available triangles (e.g. a graph leaf) - such edges are kept
//!   as-is (unverifiable, not assumed bad), and a 3-image component only
//!   ever has one triangle per edge, so a mutually-consistent bad pair of
//!   edges has no independent triplet to catch it. Inherent limit of
//!   triplet-based filtering at small scale, not a bug to fix here.
//! - A scene with genuinely large baseline-length variation (e.g. a few
//!   very close-together shots mixed with a few far-apart ones) is
//!   under-served by the unit-baseline assumption above - the resulting
//!   initial geometry is only *directionally* correct per edge, corrected
//!   for scale only by whatever the downstream bundle adjustment can pull
//!   back into consistency from track reprojection error.

use std::collections::{HashMap, HashSet};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use sfm_core::{CameraModel, Pose, Reconstruction};
use sfm_geometry::{
    reprojection_error_normalized, to_normalized, triangulate_normalized, triangulation_angle,
};

use super::{assemble_reconstruction, keypoint_px, run_bundle_adjustment, PointWork};

pub struct GlobalParams {
    /// Extra floor on top of whatever `sfm match` already verified a pair
    /// with - a low bar by default since the match stage's own
    /// `min_inliers`/`min_inlier_ratio` gate has already run.
    pub min_pair_inliers: usize,
    pub triplet_max_loop_error_deg: f64,
    pub rotation_averaging_iterations: usize,
    pub rotation_huber_deg: f64,
    /// IRLS reweight-and-resolve rounds for translation/position averaging.
    pub translation_averaging_iterations: usize,
    /// Minimum triangulation angle for track formation, against the raw
    /// averaged (not yet bundle-adjusted) poses. Deliberately looser than
    /// `IncrementalParams::min_triangulation_angle_deg`'s default (2.0) for
    /// the same reason `track_max_reprojection_error_px` is looser than
    /// `max_reprojection_error_px`: `average_translations`'s unit-baseline
    /// assumption systematically over/under-states relative depth per
    /// edge, which the angle this computes is directly sensitive to, even
    /// for genuinely well-conditioned points.
    pub min_triangulation_angle_deg: f64,
    /// Reprojection-error gate for *initial* track formation, against the
    /// raw averaged (not yet bundle-adjusted) poses - deliberately much
    /// looser than `max_reprojection_error_px` (the final BA's own outlier
    /// filter). Position averaging only assumes a roughly-unit baseline per
    /// view-graph edge (see `average_translations`'s doc comment), so a
    /// real track's pre-BA reprojection error routinely exceeds a threshold
    /// tuned for already-refined poses even when the track is genuinely
    /// sound - closing that gap is exactly the final BA's job, not this
    /// gate's.
    pub track_max_reprojection_error_px: f64,
    pub max_reprojection_error_px: f64,
    pub ba_robust_loss: sfm_ba::RobustLoss,
    /// Defaults to `false`, unlike `IncrementalParams::refine_intrinsics`.
    /// `run_bundle_adjustment`'s existing self-calibration safeguards
    /// (`MIN_IMAGES_PER_CAMERA_FOR_INTRINSICS`, the distortion-plausibility
    /// bound, and picking whichever of the free/fixed-intrinsics outputs
    /// has lower plain reprojection error) are enough to keep the
    /// incremental pipeline's own intrinsics refinement well-behaved, but
    /// measurably let a bad joint solve through on `temple_ring` (47
    /// images) when fed this pipeline's looser, not-yet-BA-refined initial
    /// track set: mean reprojection error 409px with intrinsics refinement
    /// on vs. 0.82px with it off, on the same input - the other two test
    /// datasets were within noise either way (see decisions.md's "Global
    /// pipeline"). Track building's own thresholds are the more likely
    /// place to fix this properly; disabling intrinsics refinement by
    /// default is the safe choice until that's tightened.
    pub refine_intrinsics: bool,
}

impl Default for GlobalParams {
    fn default() -> Self {
        GlobalParams {
            min_pair_inliers: 15,
            triplet_max_loop_error_deg: 5.0,
            rotation_averaging_iterations: 25,
            rotation_huber_deg: 5.0,
            translation_averaging_iterations: 5,
            min_triangulation_angle_deg: 1.0,
            track_max_reprojection_error_px: 20.0,
            max_reprojection_error_px: 4.0,
            ba_robust_loss: sfm_ba::RobustLoss::Huber,
            refine_intrinsics: false,
        }
    }
}

/// One filtered view-graph edge, `i < j` (matches `PairInput`'s own
/// invariant). `rotation`/`translation` are `TwoViewGeometryRecord.pose`
/// verbatim: `R_j = rotation * R_i`, `translation` unit-scale.
struct Edge {
    i: usize,
    j: usize,
    rotation: UnitQuaternion<f64>,
    translation: Vector3<f64>,
    /// Survived triplet cycle-consistency filtering (or had no triplet to
    /// check it against, in which case it's kept but never `validated`).
    valid: bool,
    /// Corroborated by at least one good-loop-closure triplet.
    validated: bool,
}

fn other_end(edge: &Edge, from: usize) -> usize {
    if edge.i == from {
        edge.j
    } else {
        edge.i
    }
}

/// Rotation `R` such that `R_to = R * R_from`, regardless of which of
/// `edge.i`/`edge.j` is `from`.
fn edge_rotation(edge: &Edge, from: usize, to: usize) -> UnitQuaternion<f64> {
    if from == edge.i && to == edge.j {
        edge.rotation
    } else if from == edge.j && to == edge.i {
        edge.rotation.inverse()
    } else {
        unreachable!("edge_rotation called with endpoints not matching this edge")
    }
}

fn build_edges(input: &super::ReconstructionInput, min_pair_inliers: usize) -> Vec<Edge> {
    input
        .pairs
        .iter()
        .filter(|p| p.geometry.inlier_matches.len() >= min_pair_inliers)
        .map(|p| Edge {
            i: p.i,
            j: p.j,
            rotation: p.geometry.pose.rotation,
            translation: p.geometry.pose.translation,
            valid: true,
            validated: false,
        })
        .collect()
}

/// Marks edges contradicted by every triplet they participate in as
/// `!valid` (see the module doc comment for the small-component limits of
/// this check), and edges corroborated by at least one consistent triplet
/// as `validated` (used to prefer trustworthy edges when seeding rotation
/// averaging's spanning tree).
fn filter_by_triplet_consistency(edges: &mut [Edge], n: usize, max_loop_error_deg: f64) {
    let mut neighbor_sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut lookup: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, e) in edges.iter().enumerate() {
        neighbor_sets[e.i].insert(e.j);
        neighbor_sets[e.j].insert(e.i);
        lookup.insert((e.i, e.j), idx);
    }

    let max_loop_error_rad = max_loop_error_deg.to_radians();
    let mut good = vec![0usize; edges.len()];
    let mut bad = vec![0usize; edges.len()];

    for e_idx in 0..edges.len() {
        let (i, j) = (edges[e_idx].i, edges[e_idx].j);
        let common: Vec<usize> = neighbor_sets[i]
            .intersection(&neighbor_sets[j])
            .copied()
            .collect();
        for k in common {
            let e_ik = lookup[&(i.min(k), i.max(k))];
            let e_jk = lookup[&(j.min(k), j.max(k))];
            let r_ij = edge_rotation(&edges[e_idx], i, j);
            let r_jk = edge_rotation(&edges[e_jk], j, k);
            let r_ki = edge_rotation(&edges[e_ik], k, i);
            let loop_angle = (r_ki * r_jk * r_ij).angle();
            if loop_angle <= max_loop_error_rad {
                good[e_idx] += 1;
                good[e_jk] += 1;
                good[e_ik] += 1;
            } else {
                bad[e_idx] += 1;
                bad[e_jk] += 1;
                bad[e_ik] += 1;
            }
        }
    }

    for (idx, edge) in edges.iter_mut().enumerate() {
        edge.validated = good[idx] > 0;
        if bad[idx] > 0 && good[idx] == 0 {
            edge.valid = false;
        }
    }
}

fn connected_components(n: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in edges.iter().filter(|e| e.valid) {
        adj[e.i].push(e.j);
        adj[e.j].push(e.i);
    }
    let mut visited = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        components.push(comp);
    }
    components
}

/// Initial per-image rotation via a flood-filled spanning tree from `root`
/// (`R_root = identity`), preferring `validated` edges over merely `valid`
/// (unverifiable) ones when both are available - a bad, unverifiable edge
/// routed into the tree would otherwise seed an entire subtree from a wrong
/// rotation that IRLS refinement (below) can't distinguish from a genuinely
/// consistent one, since it only measures disagreement against each node's
/// own (already-poisoned) neighbors.
fn spanning_tree_rotations(
    members: &[usize],
    edges: &[Edge],
    root: usize,
) -> HashMap<usize, UnitQuaternion<f64>> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, e) in edges.iter().enumerate().filter(|(_, e)| e.valid) {
        adj.entry(e.i).or_default().push(idx);
        adj.entry(e.j).or_default().push(idx);
    }

    let mut rotations = HashMap::new();
    rotations.insert(root, UnitQuaternion::identity());

    for validated_only in [true, false] {
        let mut progressed = true;
        while progressed {
            progressed = false;
            // Sorted, not raw `HashMap::keys()` order (randomized per
            // process) - this flood-fill assigns a node to whichever
            // known frontier node reaches it *first* within a pass, so an
            // unsorted key order would make the resulting spanning tree
            // (and therefore the pipeline's numeric output) vary between
            // runs on identical input, contrary to this codebase's
            // reproducible-`sfm map` design goal (see the seeded-xorshift
            // RANSAC sampler elsewhere for the same principle).
            let mut known: Vec<usize> = rotations.keys().copied().collect();
            known.sort_unstable();
            for u in known {
                let r_u = rotations[&u];
                for &e_idx in adj.get(&u).into_iter().flatten() {
                    let edge = &edges[e_idx];
                    if validated_only && !edge.validated {
                        continue;
                    }
                    let v = other_end(edge, u);
                    if rotations.contains_key(&v) {
                        continue;
                    }
                    rotations.insert(v, edge_rotation(edge, u, v) * r_u);
                    progressed = true;
                }
            }
        }
    }

    debug_assert!(members.iter().all(|m| rotations.contains_key(m)));
    rotations
}

/// Jacobi-style IRLS refinement of the spanning-tree rotations: each pass
/// recomputes every non-root image's rotation as a Huber-weighted,
/// hemisphere-corrected average of the quaternion "votes" implied by its
/// current neighbors, from a single snapshot (all updates land in `next`,
/// committed together) so the result doesn't depend on iteration order.
/// Converges quickly *because* it starts from an already-propagated
/// spanning-tree init, not a naive zero-init - each pass only needs to
/// locally reconcile non-tree (cycle) edges, not repropagate root's
/// influence across the whole graph.
fn refine_rotations_irls(
    members: &[usize],
    edges: &[Edge],
    root: usize,
    initial: HashMap<usize, UnitQuaternion<f64>>,
    iterations: usize,
    huber_deg: f64,
) -> HashMap<usize, UnitQuaternion<f64>> {
    let huber_rad = huber_deg.to_radians();
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, e) in edges.iter().enumerate().filter(|(_, e)| e.valid) {
        adj.entry(e.i).or_default().push(idx);
        adj.entry(e.j).or_default().push(idx);
    }

    let mut current = initial;
    for _ in 0..iterations {
        let mut next = current.clone();
        for &u in members {
            if u == root {
                continue;
            }
            let cur_u = current[&u];
            let mut sum = nalgebra::Vector4::zeros();
            let mut weight_sum = 0.0_f64;
            for &e_idx in adj.get(&u).into_iter().flatten() {
                let edge = &edges[e_idx];
                let v = other_end(edge, u);
                let Some(&r_v) = current.get(&v) else {
                    continue;
                };
                let vote = edge_rotation(edge, v, u) * r_v;
                let angle = (vote * cur_u.inverse()).angle();
                let weight = if angle <= huber_rad {
                    1.0
                } else {
                    huber_rad / angle.max(1e-9)
                };
                let mut q = vote.coords;
                if q.dot(&cur_u.coords) < 0.0 {
                    q = -q;
                }
                sum += weight * q;
                weight_sum += weight;
            }
            if weight_sum > 1e-12 {
                next.insert(
                    u,
                    UnitQuaternion::new_normalize(Quaternion {
                        coords: sum / weight_sum,
                    }),
                );
            }
        }
        current = next;
    }
    current
}

/// World-frame direction from `edge.i`'s camera center to `edge.j`'s
/// (`d_ij = -R_j^T * t_ij`, unit-scale - see the crate-level derivation in
/// `run_global`'s doc comment... actually documented at the call site
/// below), given `edge.j`'s already-averaged rotation.
fn edge_direction(edge: &Edge, rotations: &HashMap<usize, UnitQuaternion<f64>>) -> Vector3<f64> {
    let r_j = rotations[&edge.j];
    -(r_j.inverse() * edge.translation)
}

/// Direction from `edge`'s `from` endpoint to its other endpoint (either
/// orientation), derived from [`edge_direction`] (which is always stated
/// `edge.i -> edge.j`).
fn direction_from(
    edge: &Edge,
    from: usize,
    rotations: &HashMap<usize, UnitQuaternion<f64>>,
) -> Vector3<f64> {
    let d = edge_direction(edge, rotations);
    if edge.i == from {
        d
    } else {
        -d
    }
}

/// Estimates camera centers `{C_i}` for every image in the component.
///
/// A first version solved a single global weighted-least-squares system on
/// `skew(d_ij)·(C_j - C_i) ≈ 0` (direction-only constraints, gauge-fixed by
/// `C_root = 0` plus one hard-fixed "anchor" neighbor at unit distance).
/// That formulation turned out to be **badly conditioned on real, noisy
/// data**: every edge directly touching `root` (pinned at the literal
/// origin) is *exactly* scale-invariant in this system regardless of which
/// other point is nominally fixed, and since `root` is chosen as the
/// highest-degree image (many direct neighbors, by design - see
/// `refine_rotations_irls`), most of the graph had no real force resisting
/// collapse toward the origin. Measured on real test data: recovered
/// camera centers clustered within a fraction of the fixed anchor's own
/// distance from root, and ~95% of tracks then failed the triangulation-
/// angle gate downstream. Removing the anchor constraint entirely (relying
/// on the least-squares system's true 1-dof null space plus a
/// minimum-norm SVD solve) made it *worse*: with no point fixed away from
/// the origin, the system becomes homogeneous and its true minimum-norm
/// solution is trivially everyone-at-the-origin.
///
/// This version instead mirrors `refine_rotations_irls`'s own successful
/// strategy - a spanning-tree init that can't collapse by construction,
/// refined locally rather than solved globally from a cold start:
/// 1. Flood-fill a spanning tree from `root` (`C_root = 0`), assuming a
///    **unit baseline per hop** (`C_child = C_parent + 1.0 * direction`).
///    Real baselines vary in length, but this can never produce a
///    degenerate/collapsed configuration, unlike a cold least-squares
///    solve - every tree edge contributes a genuine nonzero step.
/// 2. `translation_averaging_iterations` Jacobi passes: each non-root image
///    re-estimates its position as a Huber-weighted average of the "votes"
///    implied by its current neighbors, where each vote reuses the
///    *already-established* local scale (projecting the node's current
///    offset from that neighbor onto the edge's known direction) rather
///    than reintroducing a fresh, ungrounded scale unknown - this improves
///    multi-edge consistency without ever solving the poorly-conditioned
///    global system that caused the collapse above.
///
/// A node touched by only one edge simply keeps its spanning-tree position
/// (no second edge to refine it against) - its true baseline can't be
/// recovered from direction alone regardless of method, so the unit-
/// baseline assumption is the best available estimate, flagged in the
/// module doc comment as a known limitation.
fn average_translations(
    edges: &[Edge],
    member_set: &HashSet<usize>,
    rotations: &HashMap<usize, UnitQuaternion<f64>>,
    root: usize,
    iterations: usize,
) -> HashMap<usize, Vector3<f64>> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, e) in edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.valid && member_set.contains(&e.i))
    {
        adj.entry(e.i).or_default().push(idx);
        adj.entry(e.j).or_default().push(idx);
    }

    let mut centers: HashMap<usize, Vector3<f64>> = HashMap::new();
    centers.insert(root, Vector3::zeros());
    let mut progressed = true;
    while progressed {
        progressed = false;
        // Sorted for the same reproducibility reason as
        // `spanning_tree_rotations`'s own flood-fill.
        let mut known: Vec<usize> = centers.keys().copied().collect();
        known.sort_unstable();
        for u in known {
            let c_u = centers[&u];
            for &e_idx in adj.get(&u).into_iter().flatten() {
                let edge = &edges[e_idx];
                let v = other_end(edge, u);
                if centers.contains_key(&v) {
                    continue;
                }
                centers.insert(v, c_u + direction_from(edge, u, rotations));
                progressed = true;
            }
        }
    }
    debug_assert!(member_set.iter().all(|m| centers.contains_key(m)));

    for _ in 0..iterations {
        let mut next = centers.clone();
        for &u in member_set {
            if u == root {
                continue;
            }
            let cur_u = centers[&u];
            let votes: Vec<Vector3<f64>> = adj
                .get(&u)
                .into_iter()
                .flatten()
                .filter_map(|&e_idx| {
                    let edge = &edges[e_idx];
                    let v = other_end(edge, u);
                    let c_v = *centers.get(&v)?;
                    let dir = direction_from(edge, v, rotations);
                    // Preserve whatever scale is already established
                    // between u and v (project u's current offset from v
                    // onto the edge's known direction) rather than
                    // introducing a fresh, ungrounded per-edge scale.
                    let scale = (cur_u - c_v).dot(&dir);
                    let scale = if scale.abs() < 1e-9 { 1.0 } else { scale };
                    Some(c_v + scale * dir)
                })
                .collect();
            if votes.is_empty() {
                continue;
            }
            let mut residuals: Vec<f64> = votes.iter().map(|v| (v - cur_u).norm()).collect();
            residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = residuals[residuals.len() / 2].max(1e-9);
            let mut sum = Vector3::zeros();
            let mut weight_sum = 0.0_f64;
            for vote in &votes {
                let residual = (vote - cur_u).norm();
                let weight = if residual <= median {
                    1.0
                } else {
                    median / residual
                };
                sum += weight * vote;
                weight_sum += weight;
            }
            if weight_sum > 1e-9 {
                next.insert(u, sum / weight_sum);
            }
        }
        centers = next;
    }

    centers
}

/// Simple union-find (iterative path compression, union by size).
struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        }
    }
}

/// Track building: union-find over `(image, keypoint)` observations linked
/// by verified inlier matches (any pair with both endpoints in
/// `member_set`, regardless of whether that edge survived triplet
/// filtering - the 2D-2D match itself was already RANSAC-verified by
/// `sfm-match`, independent of whether the edge was trustworthy for
/// rotation averaging specifically). A union that would put two
/// observations from the same image into one track is rejected. Every
/// resulting track of length >= 2 is triangulated (N-view, from the
/// averaged absolute poses) and gated by `min_triangulation_angle_deg` /
/// `max_reprojection_error_px`, mirroring `triangulate_pair_matches`'s own
/// gating for the incremental pipeline.
fn build_tracks(
    input: &super::ReconstructionInput,
    member_set: &HashSet<usize>,
    poses: &[Option<Pose>],
    cameras: &HashMap<u32, CameraModel>,
    min_triangulation_angle_deg: f64,
    max_reprojection_error_px: f64,
) -> Vec<PointWork> {
    let mut node_of: HashMap<(usize, u32), usize> = HashMap::new();
    let mut node_obs: Vec<(usize, u32)> = Vec::new();
    let mut pairs_used: Vec<(usize, usize)> = Vec::new();

    for pair in &input.pairs {
        if !member_set.contains(&pair.i) || !member_set.contains(&pair.j) {
            continue;
        }
        for &(ka, kb) in &pair.geometry.inlier_matches {
            let na = *node_of.entry((pair.i, ka)).or_insert_with(|| {
                node_obs.push((pair.i, ka));
                node_obs.len() - 1
            });
            let nb = *node_of.entry((pair.j, kb)).or_insert_with(|| {
                node_obs.push((pair.j, kb));
                node_obs.len() - 1
            });
            pairs_used.push((na, nb));
        }
    }

    let n = node_obs.len();
    let mut dsu = DisjointSet::new(n);
    let mut images_in: HashMap<usize, HashSet<usize>> = (0..n)
        .map(|idx| {
            let mut s = HashSet::new();
            s.insert(node_obs[idx].0);
            (idx, s)
        })
        .collect();

    for (na, nb) in pairs_used {
        let (ra, rb) = (dsu.find(na), dsu.find(nb));
        if ra == rb {
            continue;
        }
        if images_in[&ra]
            .intersection(&images_in[&rb])
            .next()
            .is_some()
        {
            continue;
        }
        let merged: HashSet<usize> = images_in[&ra].union(&images_in[&rb]).copied().collect();
        dsu.union(na, nb);
        let new_root = dsu.find(na);
        images_in.remove(&ra);
        images_in.remove(&rb);
        images_in.insert(new_root, merged);
    }

    let mut track_by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in 0..n {
        track_by_root.entry(dsu.find(idx)).or_default().push(idx);
    }

    let min_angle = min_triangulation_angle_deg.to_radians();
    let mut points = Vec::new();
    for node_indices in track_by_root.into_values() {
        if node_indices.len() < 2 {
            continue;
        }
        let track: Vec<(usize, u32)> = node_indices.iter().map(|&idx| node_obs[idx]).collect();

        let views: Vec<(Pose, (f64, f64))> = track
            .iter()
            .map(|&(img, kp)| {
                let pose = poses[img].expect("member image must have an averaged pose");
                let cam = &cameras[&input.images[img].camera_id];
                (
                    pose,
                    to_normalized(keypoint_px(&input.images[img].features, kp), cam),
                )
            })
            .collect();

        let Some(xyz) = triangulate_normalized(&views) else {
            continue;
        };

        let centers: Vec<Vector3<f64>> = views.iter().map(|(p, _)| p.camera_center()).collect();
        let mut max_angle = 0.0_f64;
        for a in 0..centers.len() {
            for b in (a + 1)..centers.len() {
                max_angle = max_angle.max(triangulation_angle(&centers[a], &centers[b], &xyz));
            }
        }
        if max_angle < min_angle {
            continue;
        }

        let mut good = true;
        for (idx, &(pose, obs)) in views.iter().enumerate() {
            let (img, _) = track[idx];
            let cam = &cameras[&input.images[img].camera_id];
            let avg_focal = (cam.focal_lengths().0 + cam.focal_lengths().1) / 2.0;
            let err_px = reprojection_error_normalized(&pose, &xyz, obs).map(|e| e * avg_focal);
            match err_px {
                Some(err) if err <= max_reprojection_error_px => {}
                _ => {
                    good = false;
                    break;
                }
            }
        }
        if !good {
            continue;
        }

        points.push(PointWork { xyz, track });
    }

    points
}

/// Entry point: view graph -> rotation averaging -> translation averaging ->
/// track building -> global bundle adjustment. Returns an empty
/// `Reconstruction` if the view graph has no component of at least 2
/// images (nothing to reconstruct).
pub fn run_global(input: &super::ReconstructionInput, params: &GlobalParams) -> Reconstruction {
    let n = input.images.len();
    if n < 2 {
        return Reconstruction::new();
    }

    let mut edges = build_edges(input, params.min_pair_inliers);
    filter_by_triplet_consistency(&mut edges, n, params.triplet_max_loop_error_deg);

    let components = connected_components(n, &edges);
    let Some(largest) = components.into_iter().max_by_key(|c| c.len()) else {
        return Reconstruction::new();
    };
    if largest.len() < 2 {
        return Reconstruction::new();
    }
    let member_set: HashSet<usize> = largest.iter().copied().collect();

    let mut degree: HashMap<usize, usize> = HashMap::new();
    for e in edges
        .iter()
        .filter(|e| e.valid && member_set.contains(&e.i))
    {
        *degree.entry(e.i).or_insert(0) += 1;
        *degree.entry(e.j).or_insert(0) += 1;
    }
    let root = *largest
        .iter()
        .max_by_key(|&&m| degree.get(&m).copied().unwrap_or(0))
        .unwrap();

    let initial_rotations = spanning_tree_rotations(&largest, &edges, root);
    let rotations = refine_rotations_irls(
        &largest,
        &edges,
        root,
        initial_rotations,
        params.rotation_averaging_iterations,
        params.rotation_huber_deg,
    );
    let centers = average_translations(
        &edges,
        &member_set,
        &rotations,
        root,
        params.translation_averaging_iterations,
    );

    // Derivation (pose convention `X_cam = R*X_world + t`, camera center
    // `C = -R^T*t`): for edge (i,j) with i<j, the stored relative pose is
    // literally what image j's pose would be if image i were the world
    // frame, giving `R_j = R_ij*R_i` and, via camera centers,
    // `C_j - C_i = -R_j^T*t_ij` - exactly `average_translations`'s
    // `edge_direction`. Recovering `t_i = -R_i*C_i` here inverts that.
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    let mut registered = vec![false; n];
    for &m in &largest {
        let (Some(&r), Some(&c)) = (rotations.get(&m), centers.get(&m)) else {
            continue;
        };
        poses[m] = Some(Pose::from_rotation_translation(r, -(r * c)));
        registered[m] = true;
    }

    let mut cameras: HashMap<u32, CameraModel> = input
        .cameras
        .iter()
        .map(|(id, cam)| (*id, cam.model))
        .collect();
    let mut points = build_tracks(
        input,
        &member_set,
        &poses,
        &cameras,
        params.min_triangulation_angle_deg,
        params.track_max_reprojection_error_px,
    );

    // Mirrors `run_incremental`'s exact two-pass final-BA pattern: a
    // fixed-intrinsics cleanup pass first, then the intrinsics-eligible
    // pass (with its built-in outlier filter-and-reoptimize loop). This
    // reuse is gauge-sound *because* `poses[root]` is exactly
    // `Pose::identity()` by construction above (`R_root = identity`,
    // `C_root = 0`): `BaInput::fixed_poses` only removes 6 of the 7 gauge
    // dof (rotation+translation, not scale); scale is instead inherited
    // from - and never perturbed away from - the anchor sitting at the
    // literal origin, since a scene rescale about the anchor's own
    // position leaves every other camera's reprojected pixels unchanged.
    // The incremental pipeline relies on this identical mechanism via its
    // own seed pose.
    run_bundle_adjustment(
        input,
        &mut cameras,
        params.ba_robust_loss,
        params.max_reprojection_error_px,
        root,
        &registered,
        &mut poses,
        &mut points,
        crate::IntrinsicsMode::Fixed,
        crate::BaScope::Global,
    );
    run_bundle_adjustment(
        input,
        &mut cameras,
        params.ba_robust_loss,
        params.max_reprojection_error_px,
        root,
        &registered,
        &mut poses,
        &mut points,
        if params.refine_intrinsics {
            crate::IntrinsicsMode::FreeGuarded
        } else {
            crate::IntrinsicsMode::Fixed
        },
        crate::BaScope::Global,
    );

    assemble_reconstruction(input, &cameras, &registered, &poses, &points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageInput, PairInput, ReconstructionInput};
    use sfm_core::{Camera, Descriptors, FeatureSet, Keypoint, TwoViewGeometryRecord};

    fn pinhole(f: f64, w: u32, h: u32) -> Camera {
        Camera {
            camera_id: 1,
            model: CameraModel::SimplePinhole {
                f,
                cx: w as f64 / 2.0,
                cy: h as f64 / 2.0,
            },
            width: w,
            height: h,
        }
    }

    fn pose_from_center(rotation: UnitQuaternion<f64>, center: Vector3<f64>) -> Pose {
        Pose::from_rotation_translation(rotation, -(rotation * center))
    }

    /// Unlike `lib.rs`'s own `relative_pose` test helper (which hands the
    /// incremental pipeline a metric-scaled fake, since it just adopts the
    /// seed pair's geometry directly as an absolute pose), `run_global`
    /// genuinely only trusts the *direction* of a pair's relative
    /// translation (see `edge_direction`) - so this normalizes to unit
    /// scale, matching what real `TwoViewGeometryRecord.pose.translation`
    /// actually looks like.
    fn relative_pose(true_poses: &[Pose], i: usize, j: usize) -> Pose {
        let ri_inv = true_poses[i].rotation.inverse();
        let rotation = true_poses[j].rotation * ri_inv;
        let translation =
            (true_poses[j].translation - rotation * true_poses[i].translation).normalize();
        Pose::from_rotation_translation(rotation, translation)
    }

    fn synthetic_points() -> Vec<Vector3<f64>> {
        (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new(
                    0.5 * (t * 0.37).sin(),
                    0.4 * (t * 0.53).cos(),
                    4.0 + 0.08 * t,
                )
            })
            .collect()
    }

    fn make_image(idx: usize, pose: &Pose, cam: &Camera, points: &[Vector3<f64>]) -> ImageInput {
        let mut keypoints = Vec::with_capacity(points.len());
        for point in points {
            let (x, y) = cam.model.project(&pose.transform_point(point));
            keypoints.push(Keypoint {
                x: x as f32,
                y: y as f32,
                scale: 1.0,
                angle: 0.0,
                response: 1.0,
            });
        }
        let n = keypoints.len();
        ImageInput {
            image_id: (idx + 1) as u32,
            camera_id: 1,
            name: format!("img{idx}.png"),
            initial_pose: None,
            pose_fixed: false,
            features: FeatureSet {
                keypoints,
                descriptors: Descriptors::Float32 {
                    dim: 1,
                    data: vec![0.0; n],
                },
            },
        }
    }

    #[test]
    fn global_pipeline_recovers_synthetic_multiview_scene() {
        let cam = pinhole(750.0, 640, 480);
        let r0 = UnitQuaternion::identity();
        let r1 = UnitQuaternion::from_euler_angles(0.0, 0.15, 0.0);
        let r2 = UnitQuaternion::from_euler_angles(0.05, 0.25, -0.02);
        let r3 = UnitQuaternion::from_euler_angles(-0.03, -0.1, 0.01);
        let r4 = UnitQuaternion::from_euler_angles(0.02, -0.18, 0.04);
        let c0 = Vector3::new(0.0, 0.0, 0.0);
        // Exactly unit distance from c0: matches `run_global`'s fixed
        // anchor-scale gauge exactly, so recovered centers can be compared
        // directly against ground truth with no alignment step needed.
        let c1 = Vector3::new(1.0, 0.0, 0.0);
        let c2 = Vector3::new(0.5, 0.9, 0.1);
        let c3 = Vector3::new(-0.4, 0.3, 0.8);
        let c4 = Vector3::new(0.2, -0.7, 0.5);
        let true_poses = vec![
            pose_from_center(r0, c0),
            pose_from_center(r1, c1),
            pose_from_center(r2, c2),
            pose_from_center(r3, c3),
            pose_from_center(r4, c4),
        ];
        let true_points = synthetic_points();

        let images: Vec<ImageInput> = true_poses
            .iter()
            .enumerate()
            .map(|(idx, pose)| make_image(idx, pose, &cam, &true_points))
            .collect();

        // Sparse-but-connected graph, deliberately not exhaustive:
        // - image 0 gets edges to 1, 2, 3, 4 (degree 4, uniquely highest ->
        //   deterministically becomes rotation-averaging root); every other
        //   image has degree >= 2 (1:{0,2}, 2:{0,1,3}, 3:{0,2,4}, 4:{0,3}) -
        //   translation averaging's cross-product-only formulation can't
        //   fix a degree-1 node's distance along its one edge (same reason
        //   a single ray can't triangulate a point's depth - see
        //   `average_translations`'s doc comment), so this graph
        //   deliberately avoids that case to keep this test's tight
        //   ground-truth tolerances meaningful.
        // - (0,1) gets every point matched (40), every other edge one
        //   fewer (39), so (0,1) uniquely wins as root's most-inlier-
        //   corroborated neighbor -> deterministically becomes the
        //   scale-gauge anchor.
        // - every edge has at least one triangle to validate it against
        //   (e.g. (0,1,2), (0,2,3), (0,3,4)), exercising triplet
        //   cycle-consistency filtering's "good" path.
        let edge_specs: [(usize, usize, usize); 7] = [
            (0, 1, 40),
            (0, 2, 39),
            (0, 3, 39),
            (0, 4, 39),
            (1, 2, 39),
            (2, 3, 39),
            (3, 4, 39),
        ];
        let pairs: Vec<PairInput> = edge_specs
            .iter()
            .map(|&(i, j, count)| {
                let matches: Vec<(u32, u32)> = (0..count as u32).map(|k| (k, k)).collect();
                PairInput {
                    i,
                    j,
                    geometry: TwoViewGeometryRecord {
                        pose: relative_pose(&true_poses, i, j),
                        inlier_matches: matches,
                    },
                }
            })
            .collect();

        let mut cameras = HashMap::new();
        cameras.insert(1u32, cam);
        let input = ReconstructionInput {
            images,
            cameras,
            pairs,
            fixed_cameras: Default::default(),
        };

        let recon = run_global(&input, &GlobalParams::default());

        assert_eq!(
            recon.images.len(),
            5,
            "expected all 5 synthetic images to register"
        );
        assert!(
            recon.points3d.len() >= 30,
            "expected most points to survive triangulation, got {}",
            recon.points3d.len()
        );

        // Rotations converge essentially exactly (angle averaging is a
        // proper least-squares fit); camera centers and points are
        // expected to carry a small, *systematic* scale offset (a few
        // percent of scene depth in this synthetic scene) rather than
        // near-machine-precision agreement - `average_translations`
        // deliberately assumes a roughly-unit baseline per view-graph edge
        // rather than recovering each edge's true relative scale (see its
        // doc comment), so even noiseless synthetic data doesn't converge
        // to an exact scale match the way the incremental pipeline's own
        // seed-anchored test does. Bundle adjustment can't correct this
        // either: reprojection error has zero gradient along the "rescale
        // the whole scene around the anchored root" direction by
        // construction (see `run_global`'s doc comment).
        for image in recon.images.values() {
            let true_idx = (image.id - 1) as usize;
            let true_pose = &true_poses[true_idx];
            let center_err = (image.pose.camera_center() - true_pose.camera_center()).norm();
            assert!(
                center_err < 0.1,
                "image {} camera center off by {center_err}",
                image.id
            );
            let angle_err = image.pose.rotation.angle_to(&true_pose.rotation);
            assert!(
                angle_err < 0.01,
                "image {} rotation off by {angle_err} rad",
                image.id
            );
        }

        for point in recon.points3d.values() {
            let te = &point.track[0];
            let true_idx = te.point2d_idx as usize;
            let err = (point.xyz - true_points[true_idx]).norm();
            assert!(err < 0.2, "point {} off by {err}", point.id);
        }
    }

    #[test]
    fn global_pipeline_only_registers_the_largest_component() {
        let cam = pinhole(750.0, 640, 480);
        // Cluster A: images 0,1,2 (fully connected). Cluster B: images 3,4
        // (one edge). No edges between the clusters.
        let poses = vec![
            pose_from_center(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
            pose_from_center(
                UnitQuaternion::from_euler_angles(0.0, 0.15, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
            ),
            pose_from_center(
                UnitQuaternion::from_euler_angles(0.05, 0.2, 0.0),
                Vector3::new(0.5, 0.8, 0.1),
            ),
            pose_from_center(UnitQuaternion::identity(), Vector3::new(20.0, 0.0, 0.0)),
            pose_from_center(
                UnitQuaternion::from_euler_angles(0.0, 0.1, 0.0),
                Vector3::new(21.0, 0.0, 0.0),
            ),
        ];
        let points = synthetic_points();
        let images: Vec<ImageInput> = poses
            .iter()
            .enumerate()
            .map(|(idx, pose)| make_image(idx, pose, &cam, &points))
            .collect();

        let edge_specs: [(usize, usize); 4] = [(0, 1), (0, 2), (1, 2), (3, 4)];
        let pairs: Vec<PairInput> = edge_specs
            .iter()
            .map(|&(i, j)| {
                let matches: Vec<(u32, u32)> = (0..points.len() as u32).map(|k| (k, k)).collect();
                PairInput {
                    i,
                    j,
                    geometry: TwoViewGeometryRecord {
                        pose: relative_pose(&poses, i, j),
                        inlier_matches: matches,
                    },
                }
            })
            .collect();

        let mut cameras = HashMap::new();
        cameras.insert(1u32, cam);
        let input = ReconstructionInput {
            images,
            cameras,
            pairs,
            fixed_cameras: Default::default(),
        };

        let recon = run_global(&input, &GlobalParams::default());

        assert_eq!(
            recon.images.len(),
            3,
            "only the 3-image cluster should register"
        );
        for id in [1u32, 2, 3] {
            assert!(
                recon.images.contains_key(&id),
                "expected cluster-A image {id} to register"
            );
        }
        for id in [4u32, 5] {
            assert!(
                !recon.images.contains_key(&id),
                "cluster-B image {id} should stay unregistered"
            );
        }
    }
}
