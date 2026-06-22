#![allow(clippy::useless_conversion)]

mod error;
mod ffi;
mod ode;
mod registry;

use ode::OdeSolver;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::ffi::c_void;

unsafe extern "C" {
    fn get_diffsol_solve_handler_f64() -> *mut c_void;
    fn get_diffsol_solve_handler_f32() -> *mut c_void;
}

static XLA_FFI_CAPSULE_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";

unsafe fn make_xla_capsule(py: Python<'_>, handler: *mut c_void) -> PyResult<PyObject> {
    let raw =
        unsafe { pyo3::ffi::PyCapsule_New(handler, XLA_FFI_CAPSULE_NAME.as_ptr().cast(), None) };
    if raw.is_null() {
        return Err(PyRuntimeError::new_err("PyCapsule_New returned null"));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, raw) })
}

#[gen_stub_pyfunction]
#[pyfunction]
fn _get_ffi_capsule_f64(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_handler_f64()) }
}

#[gen_stub_pyfunction]
#[pyfunction]
fn _get_ffi_capsule_f32(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { make_xla_capsule(py, get_diffsol_solve_handler_f32()) }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn diffsol_jax_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OdeSolver>()?;
    m.add_function(wrap_pyfunction!(_get_ffi_capsule_f64, m)?)?;
    m.add_function(wrap_pyfunction!(_get_ffi_capsule_f32, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
