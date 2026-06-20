// The #[pyfunction] macro expansion in pyo3 0.22 trips clippy::useless_conversion
// on recent clippy; the generated code is correct, so allow it crate-wide.
#![allow(clippy::useless_conversion)]

use diffsol_c::{
    HostArray, JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::os::raw::c_char;

// Getter function defined in wrapper.cc that returns the XLA FFI handler pointer.
extern "C" {
    fn get_diffsol_solve_handler() -> *mut c_void;
}

// Maps Python method codes to OdeSolverType.
// Python codes: 0=bdf, 1=tsit45, 2=esdirk34, 3=tr_bdf2
fn ode_solver_from_method(method: i32) -> Result<OdeSolverType, String> {
    match method {
        0 => Ok(OdeSolverType::Bdf),
        1 => Ok(OdeSolverType::Tsit45),
        2 => Ok(OdeSolverType::Esdirk34),
        3 => Ok(OdeSolverType::TrBdf2),
        _ => Err(format!("unknown method code {method}")),
    }
}

fn linspace(t0: f64, t_final: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![t_final];
    }
    (0..n)
        .map(|i| t0 + (t_final - t0) * i as f64 / (n - 1) as f64)
        .collect()
}

unsafe fn write_err(msg: &str, buf: *mut c_char, len: usize) {
    if !buf.is_null() && len > 0 {
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(len - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buf, copy_len);
            *buf.add(copy_len) = 0;
        }
    }
}

unsafe fn get_wrapper(handle: u64) -> Result<&'static OdeWrapper, String> {
    if handle == 0 {
        return Err("null handle".to_string());
    }
    Ok(unsafe { &*(handle as *const OdeWrapper) })
}

// Copy ys from a (n_state × n_times) col-major HostArray into an XLA (n_times, n_state)
// row-major buffer. The two layouts are identical in flat memory
// (element [s][t] = s + t*n_state), so this is effectively a memcpy, but we use
// as_array to stay within the public HostArray API.
fn copy_ys_to_xla(
    ys_ha: HostArray,
    ys_out: *mut f64,
    n_times: usize,
    n_state: usize,
) -> Result<(), String> {
    let view = ys_ha.as_array::<f64>().map_err(|e| e.to_string())?;
    // view shape is (n_state, n_times); view[[s, t]] gives ys at state s, time t.
    for t in 0..n_times {
        for s in 0..n_state {
            // XLA row-major (n_times, n_state): index t*n_state + s
            unsafe { *ys_out.add(t * n_state + s) = view[[s, t]] };
        }
    }
    Ok(())
}

fn copy_ts_to_xla(ts_ha: HostArray, ts_out: *mut f64, n_times: usize) -> Result<(), String> {
    let slice = ts_ha.as_slice::<f64>().map_err(|e| e.to_string())?;
    if slice.len() != n_times {
        return Err(format!(
            "ts length mismatch: expected {n_times}, got {}",
            slice.len()
        ));
    }
    unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), ts_out, n_times) };
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Extern "C" bridge function called from wrapper.cc
// ─────────────────────────────────────────────────────────────────────────────

/// Primal dense solve at evenly-spaced evaluation times.
///
/// This is the only Rust/FFI solve entry point. Derivatives are computed entirely
/// on the JAX side via forward sensitivity: the augmented `[y; S]` system is built
/// in Python, lowered to DiffSL, and solved through this same primal path (see
/// `python/diffsol_jax/__init__.py`). Cranelift (no LLVM/Enzyme) is sufficient.
///
/// # Safety
///
/// `handle` must be a live `OdeWrapper` pointer from `OdeSolver::handle`.
/// `params_ptr` must point to `n_params` readable `f64`s; `ys_out`/`ts_out` must
/// point to writable buffers of `n_times * n_state` and `n_times` `f64`s
/// respectively; `err_buf` must be writable for `err_buf_len` bytes (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solve_rust(
    handle: u64,
    params_ptr: *const f64,
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
    let result = (|| -> Result<(), String> {
        let wrapper = unsafe { get_wrapper(handle) }?;
        wrapper
            .set_ode_solver(ode_solver_from_method(method)?)
            .map_err(|e| e.to_string())?;

        let t_eval = linspace(t0, t_final, n_times);
        let params_ha = HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
        let t_eval_ha = HostArray::new_vector(t_eval.as_ptr() as *mut u8, n_times, ScalarType::F64);

        let solution = wrapper
            .solve_dense(params_ha, t_eval_ha)
            .map_err(|e| e.to_string())?;

        copy_ys_to_xla(
            solution.get_ys().map_err(|e| e.to_string())?,
            ys_out,
            n_times,
            n_state,
        )?;
        copy_ts_to_xla(
            solution.get_ts().map_err(|e| e.to_string())?,
            ts_out,
            n_times,
        )?;
        Ok(())
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            unsafe { write_err(&e, err_buf, err_buf_len) };
            -1
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 module
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass]
pub struct OdeSolver {
    wrapper: OdeWrapper,
}

#[pymethods]
impl OdeSolver {
    #[new]
    fn new(diffsl_src: &str) -> PyResult<Self> {
        let wrapper = OdeWrapper::new_jit(
            diffsl_src,
            JitBackendType::Cranelift,
            ScalarType::F64,
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("build: {e}")))?;

        wrapper
            .set_rtol(1e-8)
            .map_err(|e| PyRuntimeError::new_err(format!("set_rtol: {e}")))?;
        wrapper
            .set_atol(1e-8)
            .map_err(|e| PyRuntimeError::new_err(format!("set_atol: {e}")))?;

        Ok(Self { wrapper })
    }

    // Returns a stable u64 handle (raw pointer) to the inner OdeWrapper.
    // Valid for the lifetime of this Python object; PyO3 boxes the struct on the heap.
    fn handle(&self) -> u64 {
        &self.wrapper as *const OdeWrapper as u64
    }
}

// Capsule name expected by jax.ffi.register_ffi_target for CPU kernels.
static XLA_FFI_CAPSULE_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";

// Creates a Python capsule whose data IS the XLA_FFI_Handler* pointer (no wrapping layer).
// We use PyCapsule_New directly because pyo3's PyCapsule::new_bound stores the value inside
// a Box<CapsuleContents> and the capsule data would point to that box, not to the handler.
// JAX's register_ffi_target uses PyCapsule_GetPointer and expects the handler directly.
unsafe fn make_xla_capsule(py: Python<'_>, handler: *mut c_void) -> PyResult<PyObject> {
    let raw =
        unsafe { pyo3::ffi::PyCapsule_New(handler, XLA_FFI_CAPSULE_NAME.as_ptr().cast(), None) };
    if raw.is_null() {
        return Err(PyRuntimeError::new_err("PyCapsule_New returned null"));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, raw) })
}

#[pyfunction]
fn get_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_handler()) }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn diffsol_jax_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OdeSolver>()?;
    m.add_function(wrap_pyfunction!(get_ffi_capsule, m)?)?;
    Ok(())
}
