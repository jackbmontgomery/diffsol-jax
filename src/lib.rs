use diffsol::{
    AdjointOdeSolverMethod, CraneliftJitModule, DenseMatrix, DiffSl, LlvmModule, Matrix,
    MatrixCommon, NalgebraContext, NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder, OdeEquations,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, Vector,
};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

type SolveProblem = OdeSolverProblem<DiffSl<NalgebraMat<f64>, CraneliftJitModule>>;
type AdjointProblem = OdeSolverProblem<DiffSl<NalgebraMat<f64>, LlvmModule>>;

thread_local! {
    static SOLVE_CACHE: RefCell<HashMap<String, SolveProblem>> = RefCell::new(HashMap::new());
    static ADJOINT_CACHE: RefCell<HashMap<String, AdjointProblem>> = RefCell::new(HashMap::new());
}

#[repr(i32)]
#[derive(Copy, Clone)]
enum Method {
    Bdf = 0,
    Tsit45 = 1,
}

impl Method {
    fn from_i32(x: i32) -> Result<Self, String> {
        match x {
            0 => Ok(Self::Bdf),
            1 => Ok(Self::Tsit45),
            _ => Err(format!("unknown method code {x}")),
        }
    }
}

fn flatten_mat(mat: &NalgebraMat<f64>, n_state: usize, n_times: usize) -> Result<Vec<f64>, String> {
    if mat.nrows() != n_state {
        return Err(format!(
            "state size mismatch: got {} rows, expected {n_state}",
            mat.nrows()
        ));
    }
    if mat.ncols() != n_times {
        return Err(format!(
            "time size mismatch: got {} cols, expected {n_times}",
            mat.ncols()
        ));
    }
    let mut ys = Vec::with_capacity(n_times * n_state);
    for col in 0..n_times {
        for row in 0..n_state {
            ys.push(mat[(row, col)]);
        }
    }
    Ok(ys)
}

fn solve_inner(
    diffsl_src: &str,
    params: &[f64],
    t0: f64,
    t_final: f64,
    n_times: usize,
    n_state: usize,
    method: Method,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ts: Vec<f64> = (0..n_times)
        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n_times - 1) as f64))
        .collect();

    SOLVE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let problem = match cache.entry(diffsl_src.to_string()) {
            std::collections::hash_map::Entry::Occupied(e) => {
                let problem = e.into_mut();
                let p_vec = NalgebraVec::from_vec(params.to_vec(), NalgebraContext);
                problem.eqn.set_params(&p_vec);
                problem.t0 = t0;
                problem
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let problem = OdeBuilder::<NalgebraMat<f64>>::new()
                    .p(params.iter().copied())
                    .t0(t0)
                    .rtol(1e-8)
                    .atol([1e-8])
                    .build_from_diffsl::<CraneliftJitModule>(diffsl_src)
                    .map_err(|e| format!("build_from_diffsl: {e}"))?;
                Ok::<_, String>(e.insert(problem))
            }?,
        };

        // solve_dense returns [n_state x n_times], one column per output time
        let ys = match method {
            Method::Bdf => {
                let mut solver = problem
                    .bdf::<NalgebraLU<f64>>()
                    .map_err(|e| format!("bdf: {e}"))?;
                let mat = solver
                    .solve_dense(ts.as_slice())
                    .map_err(|e| format!("solve_dense: {e}"))?;
                flatten_mat(&mat, n_state, n_times)?
            }
            Method::Tsit45 => {
                let mut solver = problem.tsit45().map_err(|e| format!("tsit45: {e}"))?;
                let mat = solver
                    .solve_dense(ts.as_slice())
                    .map_err(|e| format!("solve_dense: {e}"))?;
                flatten_mat(&mat, n_state, n_times)?
            }
        };

        Ok((ys, ts))
    })
}

unsafe fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
    *buf.add(n) = 0;
}

#[no_mangle]
pub unsafe extern "C" fn diffsol_solve_rust(
    diffsl_src: *const c_char,
    diffsl_src_len: usize,
    params: *const f64,
    n_params: usize,
    t0: f64,
    t_final: f64,
    ys_out: *mut f64,
    ts_out: *mut f64,
    n_times: usize,
    n_state: usize,
    method: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let src_bytes = slice::from_raw_parts(diffsl_src as *const u8, diffsl_src_len);
        let src = std::str::from_utf8(src_bytes).map_err(|e| format!("utf8: {e}"))?;
        let params_slice = slice::from_raw_parts(params, n_params);
        let method = Method::from_i32(method)?;

        let (ys, ts) = solve_inner(src, params_slice, t0, t_final, n_times, n_state, method)?;

        if ys.len() != n_times * n_state {
            return Err(format!(
                "ys length {} != n_times*n_state {}",
                ys.len(),
                n_times * n_state
            ));
        }
        if ts.len() != n_times {
            return Err(format!("ts length {} != n_times {}", ts.len(), n_times));
        }

        slice::from_raw_parts_mut(ys_out, n_times * n_state).copy_from_slice(&ys);
        slice::from_raw_parts_mut(ts_out, n_times).copy_from_slice(&ts);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            write_err(err_buf, err_buf_len, &msg);
            1
        }
        Err(_) => {
            write_err(err_buf, err_buf_len, "rust panic in diffsol_solve_rust");
            2
        }
    }
}

fn adjoint_inner(
    diffsl_src: &str,
    params: &[f64],
    t0: f64,
    t_final: f64,
    n_times: usize,
    n_state: usize,
    g_ys: &[f64],
    method: Method,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ts: Vec<f64> = (0..n_times)
        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n_times - 1) as f64))
        .collect();

    // g_mat: [n_state x n_times], column i = dg/du at time i
    // g_ys is row-major [n_times x n_state]: g_ys[i * n_state + j] = dg/dy[j] at time i
    let mut g_mat = NalgebraMat::<f64>::zeros(n_state, n_times, NalgebraContext);
    for col in 0..n_times {
        for row in 0..n_state {
            g_mat.set_index(row, col, g_ys[col * n_state + row]);
        }
    }

    let n_params = params.len();

    // Cache the compiled problem across calls (same pattern as SOLVE_CACHE for the forward pass).
    // Each call borrows the cached problem, updates params, runs the full forward+adjoint pass,
    // then releases the borrow. All solver/checkpointer/adjoint lifetimes are scoped to the
    // closure so there are no cross-call lifetime conflicts.
    ADJOINT_CACHE.with(|cache| -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut cache = cache.borrow_mut();
        let problem = match cache.entry(diffsl_src.to_string()) {
            std::collections::hash_map::Entry::Occupied(e) => {
                let problem = e.into_mut();
                let p_vec = NalgebraVec::from_vec(params.to_vec(), NalgebraContext);
                problem.eqn.set_params(&p_vec);
                problem.t0 = t0;
                problem
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let problem = OdeBuilder::<NalgebraMat<f64>>::new()
                    .p(params.iter().copied())
                    .t0(t0)
                    .rtol(1e-8)
                    .atol([1e-8])
                    .build_from_diffsl::<LlvmModule>(diffsl_src)
                    .map_err(|e| format!("build: {e}"))?;
                Ok::<_, String>(e.insert(problem))
            }?,
        };

        // Each arm: forward re-run with checkpointing, then adjoint backward pass.
        // StateCommon is not re-exported from diffsol's crate root, so extraction
        // is inlined per arm rather than factored into a helper.
        match method {
            Method::Bdf => {
                let mut solver = problem
                    .bdf::<NalgebraLU<f64>>()
                    .map_err(|e| format!("bdf: {e}"))?;
                let (checkpointer, _) = solver
                    .solve_dense_with_checkpointing(ts.as_slice(), None)
                    .map_err(|e| format!("fwd checkpointing: {e}"))?;
                let adj = problem
                    .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointer, Some(1))
                    .map_err(|e| format!("adj_bdf: {e}"))?;
                let common = adj
                    .solve_adjoint_backwards_pass(ts.as_slice(), &[&g_mat])
                    .map_err(|e| format!("adj_pass: {e}"))?
                    .into_common();
                let grad_p = (0..n_params).map(|i| common.sg[0].get_index(i)).collect();
                let grad_y0 = (0..n_state).map(|i| common.s[0].get_index(i)).collect();
                Ok((grad_p, grad_y0))
            }
            Method::Tsit45 => {
                let mut solver = problem
                    .tsit45()
                    .map_err(|e| format!("tsit45: {e}"))?;
                let (checkpointer, _) = solver
                    .solve_dense_with_checkpointing(ts.as_slice(), None)
                    .map_err(|e| format!("fwd checkpointing: {e}"))?;
                let adj = problem
                    .tsit45_solver_adjoint::<_>(checkpointer, Some(1))
                    .map_err(|e| format!("adj_tsit45: {e}"))?;
                let common = adj
                    .solve_adjoint_backwards_pass(ts.as_slice(), &[&g_mat])
                    .map_err(|e| format!("adj_pass: {e}"))?
                    .into_common();
                let grad_p = (0..n_params).map(|i| common.sg[0].get_index(i)).collect();
                let grad_y0 = (0..n_state).map(|i| common.s[0].get_index(i)).collect();
                Ok((grad_p, grad_y0))
            }
        }
    })
}

#[pyfunction]
fn adjoint_smoke_test(py: Python<'_>) -> PyResult<PyObject> {
    let src = r#"
in_i { alpha = 1.5, beta = 1.0, delta = 0.75, gamma = 3.0 }
u_i { x = 1.0, y = 0.5 }
v0 { alpha * x - beta * x * y }
v1 { delta * x * y - gamma * y }
F_i {
  v0,
  v1,
}
"#;
    let params = [1.5f64, 1.0, 0.75, 3.0];
    let n_times = 50usize;
    let n_state = 2usize;
    let g_ys = vec![1.0f64; n_times * n_state];

    let (grad_p, grad_y0) =
        adjoint_inner(src, &params, 0.0, 10.0, n_times, n_state, &g_ys, Method::Bdf)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let all: Vec<f64> = grad_p.into_iter().chain(grad_y0).collect();
    Ok(all.to_object(py))
}

#[no_mangle]
pub unsafe extern "C" fn diffsol_vjp_rust(
    diffsl_src: *const c_char,
    diffsl_src_len: usize,
    params: *const f64,
    n_params: usize,
    t0: f64,
    t_final: f64,
    g_ys: *const f64,
    grad_p_out: *mut f64,
    grad_y0_out: *mut f64,
    n_times: usize,
    n_state: usize,
    method: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let src_bytes = slice::from_raw_parts(diffsl_src as *const u8, diffsl_src_len);
        let src = std::str::from_utf8(src_bytes).map_err(|e| format!("utf8: {e}"))?;
        let params_slice = slice::from_raw_parts(params, n_params);
        let g_ys_slice = slice::from_raw_parts(g_ys, n_times * n_state);
        let method = Method::from_i32(method)?;

        let (grad_p, grad_y0) =
            adjoint_inner(src, params_slice, t0, t_final, n_times, n_state, g_ys_slice, method)?;

        slice::from_raw_parts_mut(grad_p_out, n_params).copy_from_slice(&grad_p);
        slice::from_raw_parts_mut(grad_y0_out, n_state).copy_from_slice(&grad_y0);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            write_err(err_buf, err_buf_len, &msg);
            1
        }
        Err(_) => {
            write_err(err_buf, err_buf_len, "rust panic in diffsol_vjp_rust");
            2
        }
    }
}

extern "C" {
    fn DiffsolSolve();
    fn DiffsolVjp();
}

#[pyfunction]
fn get_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    // PyCapsule stores the name pointer by reference, so it must be static.
    static CAPSULE_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";
    let ptr = DiffsolSolve as *mut std::ffi::c_void;
    unsafe {
        let cap = pyo3::ffi::PyCapsule_New(ptr, CAPSULE_NAME.as_ptr() as *const c_char, None);
        if cap.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "PyCapsule_New failed",
            ));
        }
        Ok(PyObject::from_owned_ptr(py, cap))
    }
}

#[pyfunction]
fn get_vjp_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    static CAPSULE_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";
    let ptr = DiffsolVjp as *mut std::ffi::c_void;
    unsafe {
        let cap = pyo3::ffi::PyCapsule_New(ptr, CAPSULE_NAME.as_ptr() as *const c_char, None);
        if cap.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "PyCapsule_New failed",
            ));
        }
        Ok(PyObject::from_owned_ptr(py, cap))
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_ffi_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(get_vjp_ffi_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(adjoint_smoke_test, m)?)?;
    Ok(())
}
