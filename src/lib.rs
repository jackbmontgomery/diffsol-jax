use diffsol::{
    CraneliftJitModule, MatrixCommon, NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod,
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

extern "C" {
    fn DiffsolSolve();
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

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_ffi_capsule, m)?)?;
    Ok(())
}
