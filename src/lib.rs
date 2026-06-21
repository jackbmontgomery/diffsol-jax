#![allow(clippy::useless_conversion)]

mod error;
mod ffi;
mod ode;
mod registry;

use ode::OdeSolver;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;

extern "C" {
    fn get_diffsol_solve_handler() -> *mut c_void;
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
