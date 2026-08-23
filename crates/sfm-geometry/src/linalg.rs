//! Small SVD-based linear-algebra helpers shared by the fundamental/essential
//! and PnP solvers. Kept independent of nalgebra's singular-value ordering
//! convention throughout: every helper finds its own min/max index instead
//! of assuming singular values come back sorted.

use nalgebra::{DMatrix, DVector, Matrix3};

/// The right singular vector for the smallest singular value of `a`, i.e. a
/// least-squares solution to the homogeneous system `a * x = 0`. This is the
/// standard DLT trick used throughout this crate (8-point F/E, PnP).
pub fn null_space_vector(a: DMatrix<f64>) -> Option<DVector<f64>> {
    let svd = nalgebra::linalg::SVD::new(a, false, true);
    let v_t = svd.v_t?;
    let (idx, _) = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
    Some(v_t.row(idx).transpose())
}

/// Project `m` onto the nearest rank-2 matrix (used to enforce the
/// fundamental matrix's rank constraint after the linear 8-point solve).
pub fn enforce_rank2(m: Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return m,
    };
    let mut sv = svd.singular_values;
    let (min_idx, _) = sv
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    sv[min_idx] = 0.0;
    u * Matrix3::from_diagonal(&sv) * v_t
}

/// Project `m` onto the essential-matrix manifold: rank 2 with the two
/// nonzero singular values forced equal (Hartley & Zisserman, eq. 9.15).
pub fn enforce_essential_constraint(m: Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return m,
    };
    let sv = svd.singular_values;
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap());
    let avg = (sv[order[0]] + sv[order[1]]) / 2.0;
    let mut new_sv = nalgebra::Vector3::zeros();
    new_sv[order[0]] = avg;
    new_sv[order[1]] = avg;
    new_sv[order[2]] = 0.0;
    u * Matrix3::from_diagonal(&new_sv) * v_t
}
