use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = diffsol_jax::stub_info()?;
    stub.generate()?;
    Ok(())
}
