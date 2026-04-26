use std::process::Command;

fn main() {
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
        panic!(
            "XLA FFI header not found at {}; check jax installation",
            ffi_header
        );
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&jax_include)
        .file("src/wrapper.cc")
        .compile("diffsol_wrapper");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let wrapper_path = format!("{}/src/wrapper.cc", manifest_dir);
    let compile_commands = format!(
        r#"[
  {{
    "directory": "{manifest_dir}",
    "file": "{wrapper_path}",
    "arguments": [
      "clang++",
      "-std=c++17",
      "-I{jax_include}",
      "-c",
      "{wrapper_path}"
    ]
  }}
]
"#
    );
    std::fs::write(
        format!("{}/compile_commands.json", manifest_dir),
        compile_commands,
    )
    .expect("failed to write compile_commands.json");

    println!("cargo:rerun-if-changed=src/wrapper.cc");
    println!("cargo:rerun-if-changed=build.rs");
}
