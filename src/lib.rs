use std::ffi::c_void;

// Unused from Rust; makes Cargo build `diffsol-c` and link its C ABI into this cdylib.
use diffsol_c as _;

extern "C" {
    fn diffsol_jax_c_api() -> *const c_void;
}

#[no_mangle]
pub extern "C" fn diffsol_jax_get_api() -> *const c_void {
    unsafe { diffsol_jax_c_api() }
}
