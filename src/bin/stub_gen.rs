//! Emits `python/diffsol_jax/_rust.pyi` from the PyO3 stub-gen annotations.
//! Run with: `PYO3_PYTHON=.venv/bin/python cargo run --bin stub_gen`.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // Reads python-source / module-name from pyproject.toml, so the stub lands
    // next to the compiled extension at python/diffsol_jax/_rust.pyi.
    let stub = diffsol_jax::stub_info()?;
    stub.generate()?;
    Ok(())
}
