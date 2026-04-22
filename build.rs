use std::process::Command;

fn main() {
    // PYO3_PYTHON is set by maturin to the target venv python; fall back to python3.
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python)
        .args(["-c", "import jax.ffi; print(jax.ffi.include_dir())"])
        .output()
        .expect("failed to run python to find jax include dir");

    let jax_include = String::from_utf8(output.stdout).unwrap().trim().to_string();
    if jax_include.is_empty() {
        panic!("jax.ffi.include_dir() returned empty; is jax installed?");
    }

    let ffi_header = format!("{}/xla/ffi/api/ffi.h", jax_include);
    if !std::path::Path::new(&ffi_header).exists() {
        panic!("XLA FFI header not found at {}; check jax installation", ffi_header);
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&jax_include)
        .file("src/wrapper.cc")
        .compile("diffsol_wrapper");

    println!("cargo:rerun-if-changed=src/wrapper.cc");
    println!("cargo:rerun-if-changed=build.rs");
}
