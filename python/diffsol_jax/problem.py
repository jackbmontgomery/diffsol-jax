from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import _rust
from .lowering import make_diffsl_tuple
from .sensitivity import _make_aug_rhs, _make_solve
from .solver_type import OdeSolverLike, OdeSolverType


def _scalar_bits(dtype) -> int:
    """Map a JAX/NumPy float dtype to the solver scalar bit-width (32 or 64)."""
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return 32
    if dtype == jnp.float64:
        return 64
    raise TypeError(f"unsupported dtype {dtype}, expected float32 or float64")


class ODEProblem:
    """An ODE problem compiled from a Python RHS function.

    Wraps a user-supplied right-hand-side function, lowers it to a
    diffsl source string, and compiles it for use with ``jax.jit``,
    ``jax.grad``, ``jax.jvp``, and ``jax.vmap``.

    The compiled solver is exposed via ``solve``, which accepts ``params``
    and a time evaluation points. Initial conditions are fixed at construction
    time from ``y0``.

    Supported AD operations:
        * ``jax.grad`` / ``jax.vjp`` - reverse mode via the auto-transposed
          forward-sensitivity Jacobian
        * ``jax.jvp`` / ``jax.jacfwd`` - forward mode via the same Jacobian
        * ``jax.jit`` - compiled via the XLA custom call
        * ``jax.vmap`` - sequential batching

        **Not supported**: higher-order derivatives (``grad(grad(...))``) or
        gradients w.r.t. ``y0``.
    """

    solver: _rust.OdeSolver
    _handle: int
    n_state: int

    def __init__(
        self,
        rhs: Callable,
        y0: Float[Array, " state"],
        params: Float[Array, " params"],
    ):
        """Compile an ODE problem from a Python RHS function.

        Args:
            rhs: Right-hand-side function ``rhs(t, y, p) -> Float[Array, ...]``.
            y0: Initial state vector (shape determines ``n_state``). Fixed.
            params: Example parameter vector (shape determines ``n_params``).
        """
        self._rhs = rhs
        self._y0 = y0
        self._params = params
        self._scalar_bits = _scalar_bits(y0.dtype)

        diffsl_src = make_diffsl_tuple(rhs, y0, params)
        self.solver = _rust.OdeSolver(diffsl_src, self._scalar_bits)
        self._handle = self.solver.handle()

        self.n_state = len(y0)
        self.n_params = len(params)

        self._aug_solver = None
        self._aug_handle = None
        self._solve = _make_solve(self)

    def _ensure_aug_handle(self) -> int:
        """Build (once) the augmented forward-sensitivity solver and return its handle."""
        if self._aug_handle is None:
            aug_rhs = _make_aug_rhs(self._rhs, self.n_state, self.n_params)
            # The first differentiation of a problem may happen inside a jit trace.
            # make_diffsl_tuple bakes the initial values in as DiffSL literals (via
            # float()), so the build must run concretely, not on tracers -
            # ensure_compile_time_eval evaluates these ops eagerly even under jit.
            # y0_aug must be constructed inside this block too, otherwise the
            # concatenate produces a tracer when called during a jit trace.
            with jax.ensure_compile_time_eval():
                y0_aug = jnp.concatenate(
                    [
                        self._y0,
                        jnp.zeros(self.n_state * self.n_params, dtype=self._y0.dtype),
                    ]
                )
                aug_src = make_diffsl_tuple(aug_rhs, y0_aug, self._params)
            self._aug_solver = _rust.OdeSolver(aug_src, self._scalar_bits)
            self._aug_handle = self._aug_solver.handle()
        return self._aug_handle

    def solve(
        self,
        params: Float[Array, " params"],
        t_eval: Float[Array, " times"],
        ode_solver: OdeSolverLike = OdeSolverType.BDF,
        h0: float = 1.0,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> Tuple[Float[Array, " times"], Float[Array, "times state"]]:
        """Solve the ODE, returning ``(ts, ys)``.

        Args:
            params: Parameter vector, shape ``(n_params,)``
            t_eval: Points for the solution to be evaluated at, shape ``(times,)``
            ode_solver: Solver to use - an ``OdeSolverType`` or its name as a
                string (``"bdf"``, ``"tsit45"``, ``"esdirk34"``, ``"tr_bdf2"``).
                Defaults to ``OdeSolverType.BDF``.
            h0: Initial step size
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            ``ts`` the time points where the solutions has been evaluted (matches t_eval).
            ``ys`` array of the solution values at the evaluated time points.
        """
        if _scalar_bits(params.dtype) != self._scalar_bits:
            raise TypeError(
                f"params dtype {jnp.dtype(params.dtype)} does not match the "
                f"problem scalar type (float{self._scalar_bits}, fixed at "
                f"construction from y0)"
            )

        method_code = int(OdeSolverType.coerce(ode_solver))
        ys, ts = self._solve(method_code, h0, rtol, atol, params, t_eval)
        return ts, ys
