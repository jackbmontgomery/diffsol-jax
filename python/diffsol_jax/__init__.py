import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from . import _rust
from .lowering import make_diffsl_tuple

_REGISTERED = False


def _ensure_registered():
    global _REGISTERED
    if _REGISTERED:
        return
    ffi.register_ffi_target("diffsol_solve", _rust.get_ffi_capsule(), platform="cpu")
    _REGISTERED = True


def diffsol_solve(diffsl_src: str, params, t0, t_final, n_times: int, n_state: int):
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
    )
    return ys, ts


def make_diffsol_solver(
    rhs_tuple, y0, p_example, *, param_names=None, state_names=None, n_times=200
):
    """Trace rhs_tuple, emit DiffSL, return (solver, src).

    rhs_tuple(t, y, p) must return a tuple of scalars, one per state component.
    solver(params, t_final, t0=0.0) returns (ys, ts).
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    src = make_diffsl_tuple(
        rhs_tuple,
        y0=y0,
        p_example=p_example,
        param_names=param_names,
        state_names=state_names,
    )
    n_state = int(y0.shape[0]) if y0.ndim == 1 else 1

    def solver(params, t_final, t0=0.0):
        return diffsol_solve(src, params, t0, t_final, n_times, n_state)

    return solver, src
