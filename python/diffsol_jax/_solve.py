"""The solve: one function over a problem."""

import operator

import jax
import jax.numpy as jnp
import numpy as np

from . import _primitives
from ._lib import DiffsolJaxError
from ._params import F64
from ._problem import ODEProblem
from ._types import Ad, ArrayLike, PyTree


def _concrete(x):
    """`x` as a numpy array, or `None` if it is a tracer. The checks on the times are skipped
    for a tracer; the native side makes the ones it still can at run time."""
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
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 10_000,
    ad: Ad = "adjoint",
) -> PyTree:
    """Solve `problem` from the state `y0` at `t0` to `t1`.

    Differentiable with respect to both `y0` and `params`.
    Args:
        problem: an [`ODEProblem`][diffsol_jax.ODEProblem]: the equations, the layouts, and the
            diffsol solver.
        y0: the state at `t0`.
        params: the parameter values, in the problem's `params` structure, or `None` for a
            problem with no parameters.
        t0: the time of `y0`, where the solve starts.
        t1: the time the solve ends at, after `t0`.
        saveat: the times to return the solution at, `[nt]`, increasing, each in `[t0, t1]`.
            The default returns it at `t1` alone. The solve ends at the last time it returns,
            so a `saveat` that stops short of `t1` does less work than one that reaches it.
        rtol: the relative tolerance of the solver.
        atol: the absolute tolerance of the solver.
        max_steps: the most time steps the solver may take to reach the last time it
            returns. A solve that needs more fails rather than running on, which is what a
            stiff problem under a non-stiff solver, or too tight a tolerance, looks like.
            It is a budget for one solve: each solve of a `jax.vmap` gets it in full, and
            the backward solve of a gradient gets it again.
        ad: how to take the reverse-mode derivative: `adjoint`, the diffsol adjoint with
            checkpointing, or `forward`, the transpose of the diffsol forward sensitivities.

    Returns:
        `ys`, the solution in the problem's output structure (its `state`, or what `out`
        returns): each leaf is `[nt, *leaf_shape]`, with `nt = 1` for the default `saveat`. For
        a flat `state=n` and no `out`, that is one array `[nt, n]`.
    """
    if ad not in ("adjoint", "forward"):
        raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")

    max_steps = operator.index(max_steps)
    if max_steps < 1:
        raise DiffsolJaxError(f"max_steps must be at least 1, got {max_steps}")

    t0 = jnp.asarray(t0, F64)
    t1 = jnp.asarray(t1, F64)

    if t0.ndim != 0 or t1.ndim != 0:
        raise DiffsolJaxError("t0 and t1 must be scalars")

    t_eval = t1[None] if saveat is None else jnp.asarray(saveat, F64)

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
    """The checks on the times whose values are known; a `None` is a tracer. `times` is what
    `saveat` asked for, or `[t1]` by default -- and the default passes every check below once
    `t1` is after `t0`, so only an explicit `saveat` reaches their messages."""
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
