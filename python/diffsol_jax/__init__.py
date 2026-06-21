"""JAX bindings for the diffsol ODE solver.

A user writes the ODE right-hand side as a plain Python function ``rhs(t, y, p)``.
``ODEProblem`` lowers it to `DiffSL <https://martinjrobins.github.io/diffsl/>`_,
JIT-compiles it with diffsol's Cranelift backend,
and exposes a ``ODEProblem.solve`` method that is compatible with
``jax.jit``, ``jax.grad``, ``jax.jvp``/``jax.jacfwd``, and ``jax.vmap``.

Differentiation strategy:
    Cranelift gives us a fast primal solve but, unlike the LLVM/Enzyme backend, it does
    not emit the reverse/sensitivity kernels that diffsol's built-in adjoint needs. So
    all derivatives are obtained by *forward sensitivity analysis done at the JAX level*:

    * Alongside the primal RHS we build an **augmented RHS** for the state
      ``[y; S]`` where ``S = ∂y/∂p``. Its sensitivity block is
      ``dS_:,j/dt = ∂f/∂y · S_:,j + ∂f/∂p · e_j``, which we obtain column-by-column with
      ``jax.jvp`` of the user RHS at trace time. The whole augmented system lowers to a
      single DiffSL string and is solved as an ordinary primal solve on Cranelift.
    * Solving the augmented system materialises the full Jacobian
      ``J[t] = ∂y(t)/∂p`` of shape ``(n_times, n_state, n_params)``.
    * ``jax.custom_jvp`` then defines the tangent as ``dys = J · dp``. Because ``J``
      is a constant w.r.t. the linearisation, JAX transposes this contraction for free,
      giving reverse-mode (``grad``/``vjp``) without any adjoint solve.

    This means there is exactly one Rust/FFI entry point — the primal dense solve — used
    for both the value and (via the augmented system) its derivatives.

    The augmented solver is built lazily, only the first time a problem is
    differentiated, so plain forward evaluation never pays for it.
"""

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


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ffi.register_ffi_target("diffsol_solve", _rust.get_ffi_capsule(), platform="cpu")
    _REGISTERED = True


def _method_code(name: str) -> int:
    if name not in _METHOD_CODES:
        raise ValueError(f"unknown solver {name!r}; available: {sorted(_METHOD_CODES)}")
    return _METHOD_CODES[name]


def _ffi_solve(handle: int, params, t_span, n_times: int, n_state: int, method: int):
    """Primal dense solve via the XLA custom call. Returns ``(ys, ts)``."""
    _ensure_registered()
    return ffi.ffi_call(
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


# ── augmented (forward-sensitivity) RHS construction ─────────────────────────


def _as_vec(rhs: Callable) -> Callable:
    """Wrap ``rhs`` so it always returns a length-``n_state`` vector.

    The user RHS may return a tuple of scalar components or a single vector;
    ``jax.jvp`` needs a single array output, so normalise to that.
    """

    def vec(t, y, p):
        out = rhs(t, y, p)
        if isinstance(out, (tuple, list)):
            return jnp.stack(jnp.broadcast_arrays(*out))
        return jnp.asarray(out)

    return vec


def _make_aug_rhs(rhs: Callable, n_state: int, n_param: int) -> Callable:
    """Build the augmented RHS for the state ``[y; S]`` (``S`` flattened by column).

    The augmented state has length ``n_state * (1 + n_param)``: the first
    ``n_state`` entries are ``y``; the remaining entries are the sensitivity
    columns ``S_:,0, S_:,1, ...`` each of length ``n_state``. Each column's
    derivative is a ``jax.jvp`` of the user RHS in the direction
    ``(S_:,j, e_j)``.
    """
    rhs_vec = _as_vec(rhs)

    def aug(t, y_aug, p):
        y = y_aug[:n_state]
        outs = [rhs_vec(t, y, p)]
        for j in range(n_param):
            col = y_aug[n_state + j * n_state : n_state + (j + 1) * n_state]
            e_j = jnp.array([1.0 if k == j else 0.0 for k in range(n_param)])
            _, dcol = jax.jvp(lambda yy, pp: rhs_vec(t, yy, pp), (y, p), (col, e_j))
            outs.append(dcol)
        return jnp.concatenate(outs)

    return aug


# ── differentiable solver factory ───────────────────────────────────────────


def _make_solver(problem: "ODEProblem", method: int) -> Callable:
    """Return a ``custom_jvp`` solve function for one ``(problem, method)`` pair.

    The primal is the Cranelift dense solve; the JVP solves the augmented system
    to obtain ``J = ∂ys/∂p`` and contracts it with ``dp``. Reverse mode falls out
    of JAX's automatic transpose of that (linear) contraction.
    """
    n_state = problem.n_state
    n_param = problem.n_params
    n_times = problem.n_times
    n_aug = n_state * (1 + n_param)

    @jax.custom_jvp
    def solve(params, t_span):
        return _ffi_solve(problem._handle, params, t_span, n_times, n_state, method)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        params, t_span = primals
        dp, _dt_span = tangents  # t_span tangents are not propagated

        ys, ts = solve(params, t_span)

        aug_handle = problem._ensure_aug_handle()
        ys_aug, _ = _ffi_solve(aug_handle, params, t_span, n_times, n_aug, method)
        # ys_aug[:, n_state:] is S flattened by column: (n_times, n_param, n_state).
        # Transpose to J[t, state, param].
        jac = ys_aug[:, n_state:].reshape(n_times, n_param, n_state).transpose(0, 2, 1)

        dys = jnp.tensordot(jac, dp, axes=([2], [0]))
        return (ys, ts), (dys, jnp.zeros_like(ts))

    return solve


class ODEProblem:
    """An ODE problem compiled from a Python RHS function.

    Wraps a user-supplied right-hand-side function, lowers it to a
    `DiffSL <https://martinjrobins.github.io/diffsl/>`_ source string, and
    compiles it for use with ``jax.jit``,
    ``jax.grad``, ``jax.jvp``, and ``jax.vmap``.

    The compiled solver is exposed via ``solve``, which accepts ``params``
    and a time span. Initial conditions are fixed at construction time from
    ``y0``.

    Supported AD operations:
        * ``jax.grad`` / ``jax.vjp`` — reverse mode via the auto-transposed
          forward-sensitivity Jacobian
        * ``jax.jvp`` / ``jax.jacfwd`` — forward mode via the same Jacobian
        * ``jax.jit`` — compiled via the XLA custom call
        * ``jax.vmap`` — sequential batching

        Not supported: higher-order derivatives (``grad(grad(...))``), gradients
        w.r.t. ``y0`` (fixed at construction) or ``t_span``.
    """

    solver: _rust.OdeSolver
    _handle: int
    n_times: int
    n_state: int
    n_params: int

    def __init__(
        self,
        rhs: Callable,
        y0: Float[Array, " state"],
        params: Float[Array, " params"],
        n_times: int = 200,
    ):
        """Compile an ODE problem from a Python RHS function.

        Args:
            rhs: Right-hand-side function ``rhs(t, y, p) -> tuple[float, ...]``.
            y0: Initial state vector (shape determines ``n_state``). Fixed.
            params: Example parameter vector (shape determines ``n_params``).
            n_times: Number of output time points. Defaults to ``200``.
        """
        self._rhs = rhs
        self._y0 = jnp.asarray(y0, dtype=jnp.float64)

        diffsl_src = make_diffsl_tuple(rhs, y0, params)
        self.solver = _rust.OdeSolver(diffsl_src)
        self._handle = self.solver.handle()

        self.n_times = n_times
        self.n_state = len(y0)
        self.n_params = len(params)

        self._solvers: dict = {}
        self._aug_solver = None  # built lazily on first differentiation
        self._aug_handle = None

    def _ensure_aug_handle(self) -> int:
        """Build (once) the augmented forward-sensitivity solver and return its handle."""
        if self._aug_handle is None:
            aug_rhs = _make_aug_rhs(self._rhs, self.n_state, self.n_params)
            y0_aug = np.concatenate(
                [
                    np.asarray(self._y0, dtype=np.float64),
                    np.zeros(self.n_state * self.n_params),
                ]
            )
            p_example = np.zeros(self.n_params, dtype=np.float64)
            # The first differentiation of a problem may happen inside a jit trace.
            # make_diffsl_tuple bakes the initial values in as DiffSL literals (via
            # float()), so the build must run concretely, not on tracers —
            # ensure_compile_time_eval evaluates these ops eagerly even under jit.
            with jax.ensure_compile_time_eval():
                aug_src = make_diffsl_tuple(aug_rhs, y0_aug, p_example)
            self._aug_solver = _rust.OdeSolver(aug_src)
            self._aug_handle = self._aug_solver.handle()
        return self._aug_handle

    def solve(
        self,
        params: Float[Array, " params"],
        t_span: Float[Array, " 2"],
        ode_solver: ODE_SOLVER = "bdf",
    ) -> Tuple[Float[Array, " {n_times}"], Float[Array, "{n_times} state"]]:
        """Solve the ODE, returning ``(ts, ys)``.

        Args:
            params: Parameter vector, shape ``(n_params,)``, dtype ``float64``.
            t_span: ``[t0, t_final]``, shape ``(2,)``, dtype ``float64``.
            ode_solver: Solver name — ``"bdf"``, ``"tsit45"``,
                ``"esdirk34"``, or ``"tr_bdf2"``. Defaults to ``"bdf"``.

        Returns:
            Tuple of ``(ts, ys)`` where ``ts`` has shape ``(n_times,)`` and
            ``ys`` has shape ``(n_times, n_state)``.
        """
        code = _method_code(ode_solver)
        if code not in self._solvers:
            self._solvers[code] = _make_solver(self, code)
        solve_fn = self._solvers[code]

        params = jnp.asarray(params, dtype=jnp.float64)
        t_span = jnp.asarray(t_span, dtype=jnp.float64)

        ys, ts = solve_fn(params, t_span)
        return ts, ys
