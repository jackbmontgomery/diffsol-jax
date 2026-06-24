from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import ffi as jax_ffi
from jaxtyping import Array, Float

from . import _rust

_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    jax_ffi.register_ffi_target(
        "diffsol_solve_f64", _rust._get_ffi_capsule_f64(), platform="cpu"
    )

    jax_ffi.register_ffi_target(
        "diffsol_solve_f32", _rust._get_ffi_capsule_f32(), platform="cpu"
    )
    _REGISTERED = True


def _ffi_solve(
    handle: int,
    params: Float[Array, " state"],
    t_eval: Float[Array, " times"],
    n_times: int,
    n_state: int,
    method: int,
    rtol: float,
    atol: float,
):
    """Primal dense solve via the XLA custom call. Returns ``(ys, ts)``."""
    _ensure_registered()
    if params.dtype != t_eval.dtype:
        raise TypeError("Ensure params and t_span are the same type")
    elif params.dtype == jnp.float32:
        return jax_ffi.ffi_call(
            "diffsol_solve_f32",
            (
                jax.ShapeDtypeStruct((n_times, n_state), jnp.float32),
                jax.ShapeDtypeStruct((n_times,), jnp.float32),
            ),
            vmap_method="sequential",
        )(
            params,
            t_eval,
            handle=handle,
            method=method,
            rtol=rtol,
            atol=atol,
        )
    elif params.dtype == jnp.float64:
        return jax_ffi.ffi_call(
            "diffsol_solve_f64",
            (
                jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
                jax.ShapeDtypeStruct((n_times,), jnp.float64),
            ),
            vmap_method="sequential",
        )(
            params,
            t_eval,
            handle=handle,
            method=method,
            rtol=rtol,
            atol=atol,
        )
