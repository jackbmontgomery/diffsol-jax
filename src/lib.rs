use diffsol_c::{
    AdjointCheckpointWrapper, HostArray, JitBackendType, LinearSolverType, MatrixType,
    OdeSolverType, OdeWrapper, ScalarType, SolutionWrapper,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::os::raw::c_char;

// Getter functions defined in wrapper.cc that return XLA FFI handler pointers.
extern "C" {
    fn get_diffsol_solve_handler() -> *mut c_void;
    fn get_diffsol_solve_adjoint_fwd_handler() -> *mut c_void;
    fn get_diffsol_solve_adjoint_bkwd_handler() -> *mut c_void;
    fn get_diffsol_jvp_handler() -> *mut c_void;
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

// Bundles the forward solution (contains saved t_eval) and the adjoint checkpoint.
// Both are needed for the backward pass: the solution provides t_eval, the checkpoint
// provides forward trajectory data.
struct AdjointBundle {
    solution: SolutionWrapper,
    checkpoint: AdjointCheckpointWrapper,
}

// ─────────────────────────────────────────────────────────────────────────────
// Extern "C" bridge functions called from wrapper.cc
// ─────────────────────────────────────────────────────────────────────────────

/// Primal dense solve at evenly-spaced evaluation times.
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
        let params_ha =
            HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
        let t_eval_ha =
            HostArray::new_vector(t_eval.as_ptr() as *mut u8, n_times, ScalarType::F64);

        let solution = wrapper
            .solve_dense(params_ha, t_eval_ha)
            .map_err(|e| e.to_string())?;

        copy_ys_to_xla(solution.get_ys().map_err(|e| e.to_string())?, ys_out, n_times, n_state)?;
        copy_ts_to_xla(solution.get_ts().map_err(|e| e.to_string())?, ts_out, n_times)?;
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

/// Forward adjoint solve: computes ys and ts (primal outputs) and creates an
/// AdjointBundle on the heap whose pointer is written to *ckpt_out as a u64.
/// The caller is responsible for consuming this bundle exactly once via
/// diffsol_solve_adjoint_bkwd_rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solve_adjoint_fwd_rust(
    handle: u64,
    params_ptr: *const f64,
    n_params: usize,
    t0: f64,
    t_final: f64,
    ys_out: *mut f64,
    ts_out: *mut f64,
    ckpt_out: *mut u64,
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
        let params_ha =
            HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
        let t_eval_ha =
            HostArray::new_vector(t_eval.as_ptr() as *mut u8, n_times, ScalarType::F64);

        let (solution, checkpoint) = wrapper
            .solve_adjoint_fwd(params_ha, t_eval_ha)
            .map_err(|e| e.to_string())?;

        // Copy primal outputs before moving solution into the bundle.
        copy_ys_to_xla(solution.get_ys().map_err(|e| e.to_string())?, ys_out, n_times, n_state)?;
        copy_ts_to_xla(solution.get_ts().map_err(|e| e.to_string())?, ts_out, n_times)?;

        let bundle = Box::new(AdjointBundle {
            solution,
            checkpoint,
        });
        unsafe { *ckpt_out = Box::into_raw(bundle) as u64 };
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

/// Backward adjoint solve: runs the adjoint backward integration consuming the
/// AdjointBundle identified by ckpt_handle. The bundle is freed here regardless of
/// success or failure, so the XLA graph must call this exactly once per forward.
///
/// g_ys is (n_times, n_state) row-major in XLA. diffsol expects dgdu_eval as
/// (n_state, n_times) col-major, but these layouts are identical in flat memory,
/// so we reinterpret the XLA buffer as (n_state, n_times) col-major directly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solve_adjoint_bkwd_rust(
    handle: u64,
    g_ys_ptr: *const f64,
    grad_params_out: *mut f64,
    n_times: usize,
    n_state: usize,
    n_params: usize,
    ckpt_handle: u64,
    method: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    // Always take ownership so the bundle is freed even on error paths.
    let bundle = unsafe { Box::from_raw(ckpt_handle as *mut AdjointBundle) };

    let result = (|| -> Result<(), String> {
        let wrapper = unsafe { get_wrapper(handle) }?;
        wrapper
            .set_ode_solver(ode_solver_from_method(method)?)
            .map_err(|e| e.to_string())?;

        // Reinterpret XLA g_ys (n_times, n_state) row-major as (n_state, n_times)
        // col-major — identical flat layout — as required by solve_adjoint_bkwd.
        let dgdu_ha = HostArray::new_col_major(
            g_ys_ptr as *mut u8,
            n_state,
            n_times,
            1,                // row_stride_elems (col-major: row index varies fastest)
            n_state as isize, // col_stride_elems
            ScalarType::F64,
        );

        let grad_ha = wrapper
            .solve_adjoint_bkwd(&bundle.solution, &bundle.checkpoint, dgdu_ha)
            .map_err(|e| e.to_string())?;

        // grad_ha is (n_params, 1) 2D; access via as_array.
        let view = grad_ha.as_array::<f64>().map_err(|e| e.to_string())?;
        let n = view.shape()[0];
        if n != n_params {
            return Err(format!(
                "grad_params size mismatch: expected {n_params}, got {n}"
            ));
        }
        for i in 0..n_params {
            unsafe { *grad_params_out.add(i) = view[[i, 0]] };
        }
        Ok(())
    })();
    // bundle dropped here

    match result {
        Ok(()) => 0,
        Err(e) => {
            unsafe { write_err(&e, err_buf, err_buf_len) };
            -1
        }
    }
}

/// JVP (forward sensitivity): computes dys = J @ dp where J is the parameter
/// Jacobian from solve_fwd_sens, contracted with the direction dp.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_jvp_rust(
    handle: u64,
    params_ptr: *const f64,
    n_params: usize,
    t0: f64,
    t_final: f64,
    dp_ptr: *const f64,
    dys_out: *mut f64,
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
        let params_ha =
            HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
        let t_eval_ha =
            HostArray::new_vector(t_eval.as_ptr() as *mut u8, n_times, ScalarType::F64);

        let solution = wrapper
            .solve_fwd_sens(params_ha, t_eval_ha)
            .map_err(|e| e.to_string())?;

        // sens is Vec<HostArray> of length n_params; each array is (n_state, n_times).
        let sens = solution.get_sens().map_err(|e| e.to_string())?;
        if sens.len() != n_params {
            return Err(format!(
                "sens count mismatch: expected {n_params}, got {}",
                sens.len()
            ));
        }

        let dp = unsafe { std::slice::from_raw_parts(dp_ptr, n_params) };

        // Pre-compute 2D views for all sensitivity arrays to avoid repeated as_array calls.
        // Each view is (n_state, n_times); view[[s, t]] = d(ys[t,s])/d(params[i]).
        let views: Vec<_> = sens
            .iter()
            .map(|ha| ha.as_array::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        // dys_out: (n_times, n_state) row-major
        // dys[t][s] = sum_i dp[i] * sens_i[s][t]
        for t in 0..n_times {
            for s in 0..n_state {
                let acc: f64 = (0..n_params).map(|i| dp[i] * views[i][[s, t]]).sum();
                unsafe { *dys_out.add(t * n_state + s) = acc };
            }
        }
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
            JitBackendType::Llvm,
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
    let raw = unsafe {
        pyo3::ffi::PyCapsule_New(handler, XLA_FFI_CAPSULE_NAME.as_ptr().cast(), None)
    };
    if raw.is_null() {
        return Err(PyRuntimeError::new_err("PyCapsule_New returned null"));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, raw) })
}

#[pyfunction]
fn get_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_handler()) }
}

#[pyfunction]
fn get_solve_adjoint_fwd_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_adjoint_fwd_handler()) }
}

#[pyfunction]
fn get_solve_adjoint_bkwd_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_adjoint_bkwd_handler()) }
}

#[pyfunction]
fn get_jvp_ffi_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_jvp_handler()) }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn diffsol_jax_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OdeSolver>()?;
    m.add_function(wrap_pyfunction!(get_ffi_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(get_solve_adjoint_fwd_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(get_solve_adjoint_bkwd_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(get_jvp_ffi_capsule, m)?)?;
    Ok(())
}
