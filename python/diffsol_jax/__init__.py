import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from . import _rust
from .lowering import make_diffsl_tuple

_REGISTERED = False

_METHOD_CODES = {"bdf": 0, "tsit45": 1}


def _method_code(name: str) -> int:
    if name not in _METHOD_CODES:
        raise ValueError(f"unknown method {name!r}; available: {sorted(_METHOD_CODES)}")
    return _METHOD_CODES[name]


def _ensure_registered():
    global _REGISTERED
    if _REGISTERED:
        return
    ffi.register_ffi_target("diffsol_solve", _rust.get_ffi_capsule(), platform="cpu")
    ffi.register_ffi_target("diffsol_vjp", _rust.get_vjp_ffi_capsule(), platform="cpu")
    _REGISTERED = True


def diffsol_solve(
    diffsl_src: str, params, t0, t_final, n_times: int, n_state: int, method: int = 0
):
    _ensure_registered()
    params = jnp.asarray(params, dtype=jnp.float64)
    t_span = jnp.array([t0, t_final], dtype=jnp.float64)

    out_type = (
        jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
        jax.ShapeDtypeStruct((n_times,), jnp.float64),
    )
    ys, ts = ffi.ffi_call("diffsol_solve", out_type, vmap_method="sequential")(
        params,
        t_span,
        diffsl_src=diffsl_src,
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )
    return ys, ts


def _ffi_vjp(src, params, t_span, g_ys, n_times, n_state, method: int = 0):
    _ensure_registered()
    n_params = params.shape[0]
    out_type = (
        jax.ShapeDtypeStruct((n_params,), jnp.float64),
        jax.ShapeDtypeStruct((n_state,), jnp.float64),
    )
    return ffi.ffi_call("diffsol_vjp", out_type, vmap_method="sequential")(
        params,
        t_span,
        g_ys,
        diffsl_src=src,
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )


def make_diffsol_solver(
    rhs_tuple,
    y0,
    p_example,
    *,
    method="bdf",
    param_names=None,
    state_names=None,
    n_times=200,
):
    """Trace rhs_tuple, emit DiffSL, return (solver, src).

    rhs_tuple(t, y, p) must return a tuple of scalars, one per state component.
    solver(params, t_span) returns (ys, ts); supports jax.grad wrt params.

    method: one of "bdf" (default), "tsit45".
    """
    fwd_code = _method_code(method)

    y0 = jnp.asarray(y0, dtype=jnp.float64)
    src = make_diffsl_tuple(
        rhs_tuple,
        y0=y0,
        p_example=p_example,
        param_names=param_names,
        state_names=state_names,
    )
    n_state = int(y0.shape[0]) if y0.ndim == 1 else 1

    @jax.custom_vjp
    def solve(params, t_span):
        return diffsol_solve(
            src, params, t_span[0], t_span[1], n_times, n_state, method=fwd_code
        )

    def fwd(params, t_span):
        ys, ts = diffsol_solve(
            src, params, t_span[0], t_span[1], n_times, n_state, method=fwd_code
        )
        return (ys, ts), (params, t_span)

    def bwd(res, g):
        params, t_span = res
        g_ys, _ = g
        grad_params, _grad_y0 = _ffi_vjp(
            src, params, t_span, g_ys, n_times, n_state, method=fwd_code
        )
        return grad_params, jnp.zeros_like(t_span)

    solve.defvjp(fwd, bwd)
    return solve, src
