use diffsol::{
    AdjointOdeSolverMethod, CraneliftJitModule, DenseMatrix, LlvmModule, Matrix, MatrixCommon,
    NalgebraContext, NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod, OdeSolverState, Vector,
};
use pyo3::prelude::*;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

fn solve_inner(
    diffsl_src: &str,
    params: &[f64],
    t0: f64,
    t_final: f64,
    n_times: usize,
    n_state: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .p(params.iter().copied())
        .t0(t0)
        .rtol(1e-8)
        .atol([1e-8])
        .build_from_diffsl::<CraneliftJitModule>(diffsl_src)
        .map_err(|e| format!("build_from_diffsl: {e}"))?;

    let mut solver = problem
        .bdf::<NalgebraLU<f64>>()
        .map_err(|e| format!("bdf: {e}"))?;

    let ts: Vec<f64> = (0..n_times)
        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n_times - 1) as f64))
        .collect();

    // solve_dense returns [n_state x n_times], one column per output time
    let mat = solver
        .solve_dense(ts.as_slice())
        .map_err(|e| format!("solve_dense: {e}"))?;

    let got_rows = mat.nrows();
    let got_cols = mat.ncols();
    if got_rows != n_state {
        return Err(format!(
            "state size mismatch: got {got_rows} rows, expected {n_state}"
        ));
    }
    if got_cols != n_times {
        return Err(format!(
            "time size mismatch: got {got_cols} cols, expected {n_times}"
        ));
    }

    // flatten to row-major: ys[i * n_state + j] = state j at time i
    let mut ys = Vec::with_capacity(n_times * n_state);
    for col in 0..n_times {
        for row in 0..n_state {
            ys.push(mat[(row, col)]);
        }
    }

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
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let src_bytes = slice::from_raw_parts(diffsl_src as *const u8, diffsl_src_len);
        let src = std::str::from_utf8(src_bytes).map_err(|e| format!("utf8: {e}"))?;
        let params_slice = slice::from_raw_parts(params, n_params);

        let (ys, ts) = solve_inner(src, params_slice, t0, t_final, n_times, n_state)?;

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
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .p(params.iter().copied())
        .t0(t0)
        .rtol(1e-8)
        .atol([1e-8])
        .build_from_diffsl::<LlvmModule>(diffsl_src)
        .map_err(|e| format!("build: {e}"))?;

    let ts: Vec<f64> = (0..n_times)
        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n_times - 1) as f64))
        .collect();

    let mut solver = problem
        .bdf::<NalgebraLU<f64>>()
        .map_err(|e| format!("bdf: {e}"))?;

    let (checkpointer, _fwd_mat) = solver
        .solve_dense_with_checkpointing(ts.as_slice(), None)
        .map_err(|e| format!("fwd checkpointing: {e}"))?;

    // g_mat: [n_state x n_times], column i = dg/du at time i
    // g_ys is row-major [n_times x n_state]: g_ys[i * n_state + j] = dg/dy[j] at time i
    let mut g_mat = NalgebraMat::<f64>::zeros(n_state, n_times, NalgebraContext);
    for col in 0..n_times {
        for row in 0..n_state {
            g_mat.set_index(row, col, g_ys[col * n_state + row]);
        }
    }

    // nout_override=1: computing gradient of a single scalar loss
    let adj_solver = problem
        .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointer, Some(1))
        .map_err(|e| format!("adj_bdf: {e}"))?;

    let state = adj_solver
        .solve_adjoint_backwards_pass(ts.as_slice(), &[&g_mat])
        .map_err(|e| format!("adj_pass: {e}"))?;

    let common = state.into_common();

    let n_params = params.len();
    let mut grad_p = Vec::with_capacity(n_params);
    for i in 0..n_params {
        grad_p.push(common.sg[0].get_index(i));
    }
    let mut grad_y0 = Vec::with_capacity(n_state);
    for i in 0..n_state {
        grad_y0.push(common.s[0].get_index(i));
    }

    Ok((grad_p, grad_y0))
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

    let (grad_p, grad_y0) = adjoint_inner(src, &params, 0.0, 10.0, n_times, n_state, &g_ys)
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
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let src_bytes = slice::from_raw_parts(diffsl_src as *const u8, diffsl_src_len);
        let src = std::str::from_utf8(src_bytes).map_err(|e| format!("utf8: {e}"))?;
        let params_slice = slice::from_raw_parts(params, n_params);
        let g_ys_slice = slice::from_raw_parts(g_ys, n_times * n_state);

        let (grad_p, grad_y0) =
            adjoint_inner(src, params_slice, t0, t_final, n_times, n_state, g_ys_slice)?;

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
