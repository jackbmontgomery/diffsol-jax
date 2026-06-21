from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jax_ffi

from . import _rust

_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    jax_ffi.register_ffi_target(
        "diffsol_solve", _rust.get_ffi_capsule(), platform="cpu"
    )
    _REGISTERED = True


def _ffi_solve(handle: int, params, t_span, n_times: int, n_state: int, method: int):
    """Primal dense solve via the XLA custom call. Returns ``(ys, ts)``."""
    _ensure_registered()
    return jax_ffi.ffi_call(
        "diffsol_solve",
        (
            jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
            jax.ShapeDtypeStruct((n_times,), jnp.float64),
        ),
        vmap_method="sequential",
    )(
        params,
        t_span,
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )
