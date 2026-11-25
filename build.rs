use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("cpp");

    let diffsol_c_include = env::var("DEP_DIFFSOL_C_INCLUDE")
        .expect("diffsol-c did not propagate its include directory (DEP_DIFFSOL_C_INCLUDE)");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .pic(true)
        .warnings(true)
        .flag_if_supported("-Wextra")
        .include(&cpp_dir)
        .include(&manifest_dir.join("third_party"))
        .include(&diffsol_c_include)
        .file(cpp_dir.join("backend.cc"))
        .file(cpp_dir.join("callbacks.cc"))
        .file(cpp_dir.join("registry.cc"))
        .file(cpp_dir.join("ffi_ops.cc"))
        .file(cpp_dir.join("c_api.cc"))
        .compile("diffsol_jax_cpp");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "linux" || target_os == "macos" {
        println!("cargo:rustc-link-lib=dl");
    }

    println!("cargo:rerun-if-changed=cpp");
}
