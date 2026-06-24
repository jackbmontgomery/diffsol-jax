use diffsol_c::{HostArray, OdeSolverType, Scalar, ScalarType};
use num_traits::ToPrimitive;
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

fn copy_ys_to_xla<T: Scalar>(
    ys_ha: HostArray,
    ys_out: *mut T,
    n_times: usize,
    n_state: usize,
) -> Result<(), String> {
    let view = ys_ha.as_array::<T>().map_err(|e| e.to_string())?;
    for t in 0..n_times {
        for s in 0..n_state {
            unsafe { *ys_out.add(t * n_state + s) = view[[s, t]] };
        }
    }
    Ok(())
}

fn copy_ts_to_xla<T: Scalar>(
    ts_ha: HostArray,
    ts_out: *mut T,
    n_times: usize,
) -> Result<(), String> {
    let slice = ts_ha.as_slice::<T>().map_err(|e| e.to_string())?;
    if slice.len() != n_times {
        return Err(format!(
            "ts length mismatch: expected {n_times}, got {}",
            slice.len()
        ));
    }
    unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), ts_out, n_times) };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn solve_impl<T: Scalar + ToPrimitive>(
    handle: u64,
    params_ptr: *const f64,
    n_params: usize,
    t_eval_ptr: *const f64,
    n_times: usize,
    method: i32,
    rtol: f64,
    atol: f64,
    ys_out: *mut T,
    ts_out: *mut T,
) -> Result<(), String> {
    let wrapper = lookup(handle)?;
    wrapper
        .set_ode_solver(ode_solver_from_method(method)?)
        .map_err(|e| e.to_string())?;

    wrapper.set_rtol(rtol).map_err(|e| e.to_string())?;
    wrapper.set_atol(atol).map_err(|e| e.to_string())?;

    let params_ha = HostArray::new_vector(params_ptr as *mut u8, n_params, ScalarType::F64);
    let t_eval_ha = HostArray::new_vector(t_eval_ptr as *mut u8, n_times, ScalarType::F64);

    let solution = wrapper
        .solve_dense(params_ha, t_eval_ha)
        .map_err(|e| e.to_string())?;

    copy_ys_to_xla(
        solution.get_ys().map_err(|e| e.to_string())?,
        ys_out,
        n_times,
        wrapper.get_nstates().map_err(|e| e.to_string())?,
    )?;

    copy_ts_to_xla(
        solution.get_ts().map_err(|e| e.to_string())?,
        ts_out,
        n_times,
    )?;

    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solve_f64(
    handle: u64,
    params_ptr: *const f64,
    n_params: usize,
    t_eval_ptr: *const f64,
    n_times: usize,
    method: i32,
    rtol: f64,
    atol: f64,
    ys_out: *mut f64,
    ts_out: *mut f64,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    match solve_impl::<f64>(
        handle, params_ptr, n_params, t_eval_ptr, n_times, method, rtol, atol, ys_out, ts_out,
    ) {
        Ok(()) => 0,
        Err(e) => {
            unsafe { write_err(&e, err_buf, err_buf_len) };
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solve_f32(
    handle: u64,
    params_ptr: *const f64,
    n_params: usize,
    t_eval_ptr: *const f64,
    n_times: usize,
    method: i32,
    rtol: f64,
    atol: f64,
    ys_out: *mut f32,
    ts_out: *mut f32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i32 {
    match solve_impl::<f32>(
        handle, params_ptr, n_params, t_eval_ptr, n_times, method, rtol, atol, ys_out, ts_out,
    ) {
        Ok(()) => 0,
        Err(e) => {
            unsafe { write_err(&e, err_buf, err_buf_len) };
            -1
        }
    }
}
