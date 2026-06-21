//! Construction JIT-compiles the diffsol problem once. The solver is stored in
//! a process-global [`Registry`], and `handle()` returns its id for the JAX FFI
//! call to carry. The Python object owns the registry slot: dropping it
//! deregisters, so a compiled JAX function holding a stale id fails cleanly
//! rather than dereferencing freed memory.

use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::OnceLock;

use crate::registry::Registry;

static ODE_REGISTRY: OnceLock<Registry<OdeWrapper>> = OnceLock::new();

fn registry() -> &'static Registry<OdeWrapper> {
    ODE_REGISTRY.get_or_init(Registry::new)
}

/// The returned `OdeWrapper` is an `Arc` handle to a `Mutex`-guarded solver, so
/// the registry lock is not held during the solve and concurrent solves on one
/// handle serialize via diffsol-c's own internal lock.
pub(crate) fn lookup(handle: u64) -> Result<OdeWrapper, String> {
    if handle == 0 {
        return Err("null handle".to_string());
    }
    registry()
        .get(handle)
        .ok_or_else(|| format!("unknown handle {handle}"))
}

#[pyclass]
pub struct OdeSolver {
    id: u64,
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

        let id = registry().insert(wrapper);
        Ok(Self { id })
    }

    /// Stable opaque id for the inner solver, carried by the JAX FFI call.
    fn handle(&self) -> u64 {
        self.id
    }
}

impl Drop for OdeSolver {
    fn drop(&mut self) {
        registry().remove(self.id);
    }
}
