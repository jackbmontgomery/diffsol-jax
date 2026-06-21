//! All `unsafe` boundary handling lives here.
//! The handle is an opaque registry id, resolved via
//! [`crate::ode::lookup`].

use diffsol_c::{HostArray, OdeSolverType, ScalarType};
use std::os::raw::c_char;

use crate::error::write_err;
use crate::ode::lookup;

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

fn copy_ys_to_xla(
    ys_ha: HostArray,
    ys_out: *mut f64,
    n_times: usize,
    n_state: usize,
) -> Result<(), String> {
    let view = ys_ha.as_array::<f64>().map_err(|e| e.to_string())?;
    for t in 0..n_times {
        for s in 0..n_state {
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
        let wrapper = lookup(handle)?;
        wrapper
            .set_ode_solver(ode_solver_from_method(method)?)
            .map_err(|e| e.to_string())?;

        let t_eval = linspace(t0, t_final, n_times);
        let params_ha = HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
        let t_eval_ha = HostArray::new_vector(t_eval.as_ptr() as *mut u8, n_times, ScalarType::F64);

        let solution = wrapper
            .solve_dense(params_ha, t_eval_ha)
            .map_err(|e| e.to_string())?;

        copy_ys_to_xla(
            solution.get_ys().map_err(|e| e.to_string())?,
            ys_out,
            n_times,
            n_state,
        )?;
        copy_ts_to_xla(
            solution.get_ts().map_err(|e| e.to_string())?,
            ts_out,
            n_times,
        )?;
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
