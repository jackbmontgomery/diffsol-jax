import operator

import jax
import jax.numpy as jnp
import numpy as np

from . import _primitives
from ._problem import ODEProblem
from ._types import F32, F64, Ad, ArrayLike, DiffsolJaxError, PyTree

DEFAULT_TOL = {F32: 1e-4, F64: 1e-6}


def _concrete(x):
    try:
        return np.asarray(x, np.float64)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None


def solve(
    problem: ODEProblem,
    y0: PyTree,
    params: PyTree,
    t0: float,
    t1: float,
    *,
    saveat: ArrayLike | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_steps: int = 10_000,
    ad: Ad = "adjoint",
) -> PyTree:
    """Solve `problem` from the state `y0` at `t0` to `t1`.

    Differentiable with respect to both `y0` and `params`.
    Args:
        problem: an [`ODEProblem`][diffsol_jax.ODEProblem].
        y0: the state at `t0`.
        params: the parameter values, in the problem's `params` structure, or `None`.
        t0: initial time.
        t1: final time.
        saveat: the times to return the solution at, `[nt]`, increasing, each in `[t0, t1]`.
            The default returns it at `t1` alone.
        rtol: the relative tolerance of the solver. The default depends on the problem's
            dtype: `1e-6` for `float64`, `1e-4` for `float32`.
            atol: the absolute tolerance of the solver, with the same default as `rtol`.
        max_steps: the most timesteps the solver may take to reach the last time it
            returns.
        ad: how to take the reverse-mode derivative: `adjoint`, the diffsol adjoint with
            checkpointing, or `forward`, the transpose of the diffsol forward sensitivities.

    Returns:
        `ys`, the solution in the problem's output structure (its `state`, or what `out`
        returns)
    """
    if ad not in ("adjoint", "forward"):
        raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")

    default_tol = DEFAULT_TOL[problem.dtype]
    rtol = default_tol if rtol is None else rtol
    atol = default_tol if atol is None else atol

    max_steps = operator.index(max_steps)
    if max_steps < 1:
        raise DiffsolJaxError(f"max_steps must be at least 1, got {max_steps}")

    t0 = jnp.asarray(t0, problem.dtype)
    t1 = jnp.asarray(t1, problem.dtype)

    if t0.ndim != 0 or t1.ndim != 0:
        raise DiffsolJaxError("t0 and t1 must be scalars")

    t_eval = t1[None] if saveat is None else jnp.asarray(saveat, problem.dtype)

    if t_eval.ndim != 1:
        raise DiffsolJaxError("saveat must have one axis")
    if t_eval.shape[0] < 1:
        raise DiffsolJaxError("saveat must have at least one time")

    _check_times(_concrete(t0), _concrete(t1), _concrete(t_eval))

    flat = problem._pack(y0, params)
    prim_cls = _primitives.SolveAdjoint if ad == "adjoint" else _primitives.SolveForward
    prim = prim_cls(
        jax.typeof(flat),
        jax.typeof(t0),
        jax.typeof(t_eval),
        problem=problem,
        rtol=float(rtol),
        atol=float(atol),
        max_steps=max_steps,
    )
    return problem._unpack(prim(flat, t0, t_eval))


def _check_times(t0, t1, times):
    if t0 is not None and t1 is not None and t1 <= t0:
        raise DiffsolJaxError(
            f"t1 must be after t0, got t0={t0} and t1={t1}: a solve must advance in time"
        )
    if times is None:
        return
    if np.any(np.diff(times) <= 0.0):
        raise DiffsolJaxError("saveat must be increasing")
    if t0 is not None and times[0] < t0:
        raise DiffsolJaxError(
            f"saveat must start at or after t0, got saveat[0]={times[0]} and t0={t0}"
        )
    if t0 is not None and times[-1] <= t0:
        raise DiffsolJaxError(
            f"saveat must reach past t0, got saveat[-1]={times[-1]} and t0={t0}: a solve "
            "must advance in time"
        )
    if t1 is not None and times[-1] > t1:
        raise DiffsolJaxError(
            f"saveat must end at or before t1, got saveat[-1]={times[-1]} and t1={t1}"
        )
