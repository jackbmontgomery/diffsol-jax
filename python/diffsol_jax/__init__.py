from functools import partial
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi
from jaxtyping import Array, Float

from . import _rust
from .lowering import make_diffsl_tuple

jax.config.update("jax_enable_x64", True)

_REGISTERED = False

_METHOD_CODES = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}
ODE_RHS = Callable
ODE_SOLVER = Literal["bdf", "tsit45", "esdirk34", "tr_bdf2"]


def _ensure_registered():
    global _REGISTERED
    if _REGISTERED:
        return
    ffi.register_ffi_target("diffsol_solve", _rust.get_ffi_capsule(), platform="cpu")
    ffi.register_ffi_target("diffsol_vjp", _rust.get_vjp_ffi_capsule(), platform="cpu")
    _REGISTERED = True


def _method_code(name: str) -> int:
    if name not in _METHOD_CODES:
        raise ValueError(f"unknown solver {name!r}; available: {sorted(_METHOD_CODES)}")
    return _METHOD_CODES[name]


def _solve_forward(
    handle: int, params, t0, t_final, n_times: int, n_state: int, method: int = 0
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
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )
    return ys, ts


def _solve_adjoint(
    handle: int, params, t_span, g_ys, n_times, n_state, method: int = 0
):
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
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )


class ODEProblem:
    solver: _rust.OdeSolver
    _handle: int
    n_times: int
    solve: Callable[
        [Float[Array, " params"], Float[Array, " 2"]],
        Tuple[Float[Array, "{n_times}"], Float[Array, "{n_times} params"]],
    ]

    def __init__(
        self,
        rhs: Callable[
            [Float[Array, ""], Float[Array, " state"], Float[Array, " params"]],
            Float[Array, " state"],
        ],
        y0: Float[Array, " state"],
        params: Float[Array, " params"],
        n_times: int = 200,
        ode_solver: ODE_SOLVER = "bdf",
    ):

        diffsl_src = make_diffsl_tuple(rhs, y0, params)
        solver = _rust.OdeSolver(diffsl_src)

        self.solver = solver
        self._handle = solver.handle()
        self.n_times = n_times
        solver_code = _method_code(ode_solver)

        n_state = len(y0)

        @jax.custom_vjp
        def solve(
            params: Float[Array, " params"],
            t_span: Float[Array, " 2"],
        ) -> Tuple[Float[Array, "{n_times}"], Float[Array, "{n_times} params"]]:

            (ts, ys), _ = fwd(params, t_span)

            return ts, ys

        def fwd(params, t_span):
            ys, ts = _solve_forward(
                solver.handle(),
                params,
                t_span[0],
                t_span[1],
                n_times,
                n_state,
                solver_code,
            )
            return (ts, ys), (params, t_span)

        def bwd(res, g):
            params, t_span = res
            _, g_ys = g
            grad_params, _grad_y0 = _solve_adjoint(
                solver.handle(), params, t_span, g_ys, n_times, n_state, solver_code
            )
            return grad_params, jnp.zeros_like(t_span)

        solve.defvjp(fwd, bwd)
        self.solve = solve
