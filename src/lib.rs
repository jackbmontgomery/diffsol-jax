use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, DiffSl, LlvmModule, Matrix, MatrixCommon, NalgebraContext,
    NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod,
    OdeSolverProblem, OdeSolverState, SensitivitiesOdeSolverMethod, Vector,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::{Arc, Mutex};

type SolveProblem = OdeSolverProblem<DiffSl<NalgebraMat<f64>, LlvmModule>>;

#[pyclass]
pub struct OdeSolver {
    problem: Arc<Mutex<SolveProblem>>,
}

#[pymethods]
impl OdeSolver {
    #[new]
    fn new(diffsl_src: &str) -> PyResult<Self> {
        let problem = OdeBuilder::<NalgebraMat<f64>>::new()
            .rtol(1e-8)
            .atol([1e-8])
            .build_from_diffsl::<LlvmModule>(diffsl_src)
            .map_err(|e| PyRuntimeError::new_err(format!("build: {e}")))?;
        Ok(Self {
            problem: Arc::new(Mutex::new(problem)),
        })
    }

    fn handle(&self) -> u64 {
        Arc::as_ptr(&self.problem) as u64
    }
}

#[repr(i32)]
#[derive(Copy, Clone)]
enum Method {
    Bdf = 0,
    Tsit45 = 1,
    Esdirk34 = 2,
    TrBdf2 = 3,
}

impl Method {
    fn from_i32(x: i32) -> Result<Self, String> {
        match x {
            0 => Ok(Self::Bdf),
            1 => Ok(Self::Tsit45),
            2 => Ok(Self::Esdirk34),
            3 => Ok(Self::TrBdf2),
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

fn make_ts(n_times: usize, t0: f64, t_final: f64) -> Vec<f64> {
    (0..n_times)
        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n_times - 1) as f64))
        .collect()
}

/// Forward solve using the stored initial condition.
fn solve_inner(
    handle: u64,
    params: &[f64],
    t0: f64,
    t_final: f64,
    n_times: usize,
    n_state: usize,
    method: Method,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ts = make_ts(n_times, t0, t_final);

    let mutex = unsafe { &*(handle as *const Mutex<SolveProblem>) };
    let mut problem = mutex.lock().unwrap();

    let params_vec = NalgebraVec::from_vec(params.to_vec(), NalgebraContext);
    problem.eqn_mut().set_params(&params_vec);

    let ys = match method {
        Method::Bdf => {
            let mut solver = problem
                .bdf::<NalgebraLU<f64>>()
                .map_err(|e| format!("bdf: {e}"))?;
            let (mat, _) = solver
                .solve_dense(ts.as_slice())
                .map_err(|e| format!("solve_dense: {e}"))?;
            flatten_mat(&mat, n_state, n_times)?
        }
        Method::Tsit45 => {
            let mut solver = problem.tsit45().map_err(|e| format!("tsit45: {e}"))?;
            let (mat, _) = solver
                .solve_dense(ts.as_slice())
                .map_err(|e| format!("solve_dense: {e}"))?;
            flatten_mat(&mat, n_state, n_times)?
        }
        Method::Esdirk34 => {
            let mut solver = problem
                .esdirk34::<NalgebraLU<f64>>()
                .map_err(|e| format!("esdirk34: {e}"))?;
            let (mat, _) = solver
                .solve_dense(ts.as_slice())
                .map_err(|e| format!("solve_dense: {e}"))?;
            flatten_mat(&mat, n_state, n_times)?
        }
        Method::TrBdf2 => {
            let mut solver = problem
                .tr_bdf2::<NalgebraLU<f64>>()
                .map_err(|e| format!("tr_bdf2: {e}"))?;
            let (mat, _) = solver
                .solve_dense(ts.as_slice())
                .map_err(|e| format!("solve_dense: {e}"))?;
            flatten_mat(&mat, n_state, n_times)?
        }
    };
    Ok((ys, ts))
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
    handle: u64,
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
        let params_slice = slice::from_raw_parts(params, n_params);
        let method = Method::from_i32(method)?;

        let (ys, ts) = solve_inner(handle, params_slice, t0, t_final, n_times, n_state, method)?;

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

/// Adjoint (reverse-mode) pass using the stored initial condition.
fn adjoint_inner(
    handle: u64,
    params: &[f64],
    t0: f64,
    t_final: f64,
    n_times: usize,
    n_state: usize,
    g_ys: &[f64],
    method: Method,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ts = make_ts(n_times, t0, t_final);

    let mut g_mat = NalgebraMat::<f64>::zeros(n_state, n_times, NalgebraContext);
    for col in 0..n_times {
        for row in 0..n_state {
            g_mat.set_index(row, col, g_ys[col * n_state + row]);
        }
    }

    let n_params = params.len();

    let mutex = unsafe { &*(handle as *const Mutex<SolveProblem>) };
    let mut problem = mutex.lock().unwrap();

    let params_vec = NalgebraVec::from_vec(params.to_vec(), NalgebraContext);
    problem.eqn_mut().set_params(&params_vec);

    match method {
        Method::Bdf => {
            let mut solver = problem
                .bdf::<NalgebraLU<f64>>()
                .map_err(|e| format!("bdf: {e}"))?;
            let (checkpointer, _, _) = solver
                .solve_dense_with_checkpointing(ts.as_slice(), None)
                .map_err(|e| format!("fwd checkpointing: {e}"))?;
            let adj = problem
                .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointer, Some(1))
                .map_err(|e| format!("adj_bdf: {e}"))?;
            let common = adj
                .solve_adjoint_backwards_pass(None, ts.as_slice(), &[&g_mat])
                .map_err(|e| format!("adj_pass: {e}"))?
                .into_common();
            let grad_p = (0..n_params).map(|i| common.sg[0].get_index(i)).collect();
            let grad_y0 = (0..n_state).map(|i| common.s[0].get_index(i)).collect();
            Ok((grad_p, grad_y0))
        }
        Method::Tsit45 => {
            let mut solver = problem.tsit45().map_err(|e| format!("tsit45: {e}"))?;
            let (checkpointer, _, _) = solver
                .solve_dense_with_checkpointing(ts.as_slice(), None)
                .map_err(|e| format!("fwd checkpointing: {e}"))?;
            let adj = problem
                .tsit45_solver_adjoint::<_>(checkpointer, Some(1))
                .map_err(|e| format!("adj_tsit45: {e}"))?;
            let common = adj
                .solve_adjoint_backwards_pass(None, ts.as_slice(), &[&g_mat])
                .map_err(|e| format!("adj_pass: {e}"))?
                .into_common();
            let grad_p = (0..n_params).map(|i| common.sg[0].get_index(i)).collect();
            let grad_y0 = (0..n_state).map(|i| common.s[0].get_index(i)).collect();
            Ok((grad_p, grad_y0))
        }
        Method::Esdirk34 | Method::TrBdf2 => {
            let mut solver = problem
                .bdf::<NalgebraLU<f64>>()
                .map_err(|e| format!("bdf (adjoint fallback): {e}"))?;
            let (checkpointer, _, _) = solver
                .solve_dense_with_checkpointing(ts.as_slice(), None)
                .map_err(|e| format!("fwd checkpointing: {e}"))?;
            let adj = problem
                .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointer, Some(1))
                .map_err(|e| format!("adj_bdf: {e}"))?;
            let common = adj
                .solve_adjoint_backwards_pass(None, ts.as_slice(), &[&g_mat])
                .map_err(|e| format!("adj_pass: {e}"))?
                .into_common();
            let grad_p = (0..n_params).map(|i| common.sg[0].get_index(i)).collect();
            let grad_y0 = (0..n_state).map(|i| common.s[0].get_index(i)).collect();
            Ok((grad_p, grad_y0))
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn diffsol_vjp_rust(
    handle: u64,
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
        let params_slice = slice::from_raw_parts(params, n_params);
        let g_ys_slice = slice::from_raw_parts(g_ys, n_times * n_state);
        let method = Method::from_i32(method)?;

        let (grad_p, grad_y0) = adjoint_inner(
            handle,
            params_slice,
            t0,
            t_final,
            n_times,
            n_state,
            g_ys_slice,
            method,
        )?;

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

/// Forward-mode sensitivity: dys = J_params * dp.
///
/// Uses a single sensitivity solve (Run A) with zero initial conditions for
/// all sensitivity directions, then contracts with dp.  No dy0 term because
/// y0 is the stored initial condition and is not a traced input.
fn jvp_inner(
    handle: u64,
    params: &[f64],
    t0: f64,
    t_final: f64,
    dp: &[f64],
    n_times: usize,
    n_state: usize,
    _method: Method,
) -> Result<Vec<f64>, String> {
    let ts = make_ts(n_times, t0, t_final);

    let mutex = unsafe { &*(handle as *const Mutex<SolveProblem>) };
    let mut problem = mutex.lock().unwrap();

    let params_vec = NalgebraVec::from_vec(params.to_vec(), NalgebraContext);
    problem.eqn_mut().set_params(&params_vec);

    // Single sensitivity solve: s_i(0) = 0 for all param directions i.
    let state = problem
        .bdf_state_sens::<NalgebraLU<f64>>()
        .map_err(|e| format!("jvp bdf_state_sens: {e}"))?;
    let mut solver = problem
        .bdf_solver_sens::<NalgebraLU<f64>>(state)
        .map_err(|e| format!("jvp bdf_solver_sens: {e}"))?;
    let (_, sens, _) = solver
        .solve_dense_sensitivities(ts.as_slice())
        .map_err(|e| format!("jvp solve_dense_sensitivities: {e}"))?;

    // dys[t, s] = sum_i dp[i] * sens[i][s, t]
    let n_params = params.len();
    let mut dys = vec![0.0f64; n_times * n_state];
    for col in 0..n_times {
        for row in 0..n_state {
            let mut v = 0.0f64;
            for i in 0..n_params {
                v += dp[i] * sens[i][(row, col)];
            }
            dys[col * n_state + row] = v;
        }
    }
    Ok(dys)
}

#[no_mangle]
pub unsafe extern "C" fn diffsol_jvp_rust(
    handle: u64,
    params: *const f64,
    n_params: usize,
    t0: f64,
    t_final: f64,
    dp: *const f64,
    dys_out: *mut f64,
    n_times: usize,
    n_state: usize,
    method: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let params_slice = slice::from_raw_parts(params, n_params);
        let dp_slice = slice::from_raw_parts(dp, n_params);
        let method = Method::from_i32(method)?;

        let dys = jvp_inner(
            handle,
            params_slice,
            t0,
            t_final,
            dp_slice,
            n_times,
            n_state,
            method,
        )?;

        if dys.len() != n_times * n_state {
            return Err(format!(
                "dys length {} != n_times*n_state {}",
                dys.len(),
                n_times * n_state
            ));
        }
        slice::from_raw_parts_mut(dys_out, n_times * n_state).copy_from_slice(&dys);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            write_err(err_buf, err_buf_len, &msg);
            1
        }
        Err(_) => {
            write_err(err_buf, err_buf_len, "rust panic in diffsol_jvp_rust");
            2
        }
    }
}

extern "C" {
    fn DiffsolSolve();
    fn DiffsolVjp();
    fn DiffsolJvp();
}

#[pyfunction]
fn get_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
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

#[pyfunction]
fn get_jvp_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    static CAPSULE_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";
    let ptr = DiffsolJvp as *mut std::ffi::c_void;
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
    m.add_function(wrap_pyfunction!(get_jvp_ffi_capsule, m)?)?;
    m.add_class::<OdeSolver>()?;
    Ok(())
}
