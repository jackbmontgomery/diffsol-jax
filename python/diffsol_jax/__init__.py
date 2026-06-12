from typing import Callable, Literal, Tuple

import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jax import ffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir
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
    ffi.register_ffi_target(
        "diffsol_solve_adjoint_fwd",
        _rust.get_solve_adjoint_fwd_capsule(),
        platform="cpu",
    )
    ffi.register_ffi_target(
        "diffsol_solve_adjoint_bkwd",
        _rust.get_solve_adjoint_bkwd_capsule(),
        platform="cpu",
    )
    ffi.register_ffi_target("diffsol_jvp", _rust.get_jvp_ffi_capsule(), platform="cpu")
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


def _solve_adjoint_fwd(handle: int, params, t_span, n_times: int, n_state: int, method: int):
    """Forward adjoint solve: returns (ys, ts, ckpt_handle) where ckpt_handle is an int64
    scalar identifying the checkpoint bundle on the Rust heap."""
    _ensure_registered()
    out_type = (
        jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
        jax.ShapeDtypeStruct((n_times,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.int64),
    )
    return ffi.ffi_call(
        "diffsol_solve_adjoint_fwd",
        out_type,
        has_side_effect=True,
        vmap_method="sequential",
    )(
        params,
        t_span,
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )


def _solve_adjoint_bkwd(
    handle: int, g_ys, ckpt_handle, n_times: int, n_state: int, n_params: int, method: int
):
    """Backward adjoint solve: consumes the checkpoint bundle and returns grad_params.
    Must be called exactly once per corresponding _solve_adjoint_fwd."""
    _ensure_registered()
    out_type = jax.ShapeDtypeStruct((n_params,), jnp.float64)
    return ffi.ffi_call(
        "diffsol_solve_adjoint_bkwd",
        out_type,
        has_side_effect=True,
        vmap_method="sequential",
    )(
        g_ys,
        ckpt_handle,
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        n_params=np.int64(n_params),
        method=np.int64(method),
    )


def _solve_vjp(handle: int, params, t_span, g_ys, n_times, n_state, method: int = 0):
    """VJP via split discrete adjoint. Returns (grad_params, grad_y0_zeros).
    Provided for backward compatibility with tests; internally uses the split
    adjoint_fwd + adjoint_bkwd path."""
    _ensure_registered()
    params = jnp.asarray(params, dtype=jnp.float64)
    t_span = jnp.asarray(t_span, dtype=jnp.float64)
    g_ys = jnp.asarray(g_ys, dtype=jnp.float64)
    n_params = params.shape[0]

    _ys, _ts, ckpt = _solve_adjoint_fwd(handle, params, t_span, n_times, n_state, method)
    grad_p = _solve_adjoint_bkwd(
        handle, g_ys, ckpt, n_times, n_state, int(n_params), method
    )
    grad_y0 = jnp.zeros(n_state, dtype=jnp.float64)
    return grad_p, grad_y0


def _solve_jvp(
    handle: int, params, t0, t_final, dp, n_times: int, n_state: int, method: int = 0
):
    _ensure_registered()
    params = jnp.asarray(params, dtype=jnp.float64)
    dp = jnp.asarray(dp, dtype=jnp.float64)
    t_span = jnp.array([t0, t_final], dtype=jnp.float64)

    out_type = jax.ShapeDtypeStruct((n_times, n_state), jnp.float64)
    dys = ffi.ffi_call("diffsol_jvp", out_type, vmap_method="sequential")(
        params,
        t_span,
        dp,
        handle=np.int64(handle),
        n_times=np.int64(n_times),
        n_state=np.int64(n_state),
        method=np.int64(method),
    )
    return dys


def _make_primitive(handle: int, n_state: int, n_times: int, method: int, n_params: int):
    """Build the JAX primitive family for one (handle, method) configuration.

    Architecture
    ------------
    ``prim`` -- primal solve (no checkpoint), ``(ys, ts)``.
        Used for plain forward evaluation.

    ``prim_fwd`` -- checkpointing primal: ``(ys, ts, ckpt_handle)``.
        Used *only* under differentiation. Calls ``diffsol_solve_adjoint_fwd``.

    ``sens_prim`` -- linear primitive: ``(params, t_span, ckpt_handle, dp) → dys``.
        Its transpose calls ``diffsol_solve_adjoint_bkwd`` with the checkpoint.
    """
    _ensure_registered()

    # ── primal primitive (no checkpoint) ─────────────────────────────────────
    prim = core.Primitive(f"diffsol_{handle}_{method}")
    prim.multiple_results = True

    def prim_abstract_eval(params, t_span):
        return [
            jax_core.ShapedArray((n_times, n_state), jnp.float64),
            jax_core.ShapedArray((n_times,), jnp.float64),
        ]

    prim.def_abstract_eval(prim_abstract_eval)

    def prim_impl(params, t_span):
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

    prim.def_impl(prim_impl)

    mlir.register_lowering(
        prim,
        mlir.lower_fun(prim_impl, multiple_results=True),
        platform="cpu",
    )

    # ── checkpointing primal primitive (used under differentiation) ───────────
    prim_fwd = core.Primitive(f"diffsol_fwd_{handle}_{method}")
    prim_fwd.multiple_results = True

    def prim_fwd_abstract_eval(params, t_span):
        return [
            jax_core.ShapedArray((n_times, n_state), jnp.float64),
            jax_core.ShapedArray((n_times,), jnp.float64),
            jax_core.ShapedArray((1,), jnp.int64),  # ckpt_handle
        ]

    prim_fwd.def_abstract_eval(prim_fwd_abstract_eval)

    def prim_fwd_impl(params, t_span):
        return ffi.ffi_call(
            "diffsol_solve_adjoint_fwd",
            (
                jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
                jax.ShapeDtypeStruct((n_times,), jnp.float64),
                jax.ShapeDtypeStruct((1,), jnp.int64),
            ),
            has_side_effect=True,
            vmap_method="sequential",
        )(
            params,
            t_span,
            handle=np.int64(handle),
            n_times=np.int64(n_times),
            n_state=np.int64(n_state),
            method=np.int64(method),
        )

    prim_fwd.def_impl(prim_fwd_impl)

    mlir.register_lowering(
        prim_fwd,
        mlir.lower_fun(prim_fwd_impl, multiple_results=True),
        platform="cpu",
    )

    # ── linear sensitivity primitive ─────────────────────────────────────────
    sens_prim = core.Primitive(f"diffsol_sens_{handle}_{method}")
    sens_prim.multiple_results = False

    def sens_abstract_eval(params, t_span, ckpt_handle, dp):
        return jax_core.ShapedArray((n_times, n_state), jnp.float64)

    sens_prim.def_abstract_eval(sens_abstract_eval)

    def sens_impl(params, t_span, ckpt_handle, dp):
        # Under eager execution, run the JVP directly (no checkpoint needed).
        return ffi.ffi_call(
            "diffsol_jvp",
            jax.ShapeDtypeStruct((n_times, n_state), jnp.float64),
            vmap_method="sequential",
        )(
            params,
            t_span,
            dp,
            handle=np.int64(handle),
            n_times=np.int64(n_times),
            n_state=np.int64(n_state),
            method=np.int64(method),
        )

    sens_prim.def_impl(sens_impl)

    mlir.register_lowering(
        sens_prim,
        mlir.lower_fun(sens_impl, multiple_results=False),
        platform="cpu",
    )

    def sens_transpose(ct, params, t_span, ckpt_handle, dp):
        """VJP via the backward adjoint pass, consuming the checkpoint from prim_fwd."""
        ct_dys = ad.instantiate_zeros(ct)
        grad_p = _solve_adjoint_bkwd(
            handle, ct_dys, ckpt_handle, n_times, n_state, n_params, method
        )
        return (
            None,   # params (residual, not linear)
            None,   # t_span (residual)
            None,   # ckpt_handle (residual)
            grad_p if ad.is_undefined_primal(dp) else None,
        )

    ad.primitive_transposes[sens_prim] = sens_transpose

    def jvp_rule(vals, tans):
        params, t_span = vals
        dp, _dt_span = tans
        # Use prim_fwd to create the checkpoint for the backward pass.
        ys, ts, ckpt_handle = prim_fwd.bind(params, t_span)
        dp = ad.instantiate_zeros(dp)
        dys = sens_prim.bind(params, t_span, ckpt_handle, dp)
        return (ys, ts), (dys, jnp.zeros_like(ts))

    ad.primitive_jvps[prim] = jvp_rule

    def _batch(vals, dims, *, bind_fn, n_out):
        moved = [
            jnp.moveaxis(v, d, 0) if d is not None else v for v, d in zip(vals, dims)
        ]
        batch_size = next(v.shape[0] for v, d in zip(moved, dims) if d is not None)
        broadcast = [
            jnp.broadcast_to(v[None], (batch_size,) + v.shape) if d is None else v
            for v, d in zip(moved, dims)
        ]

        def one(*args):
            return bind_fn(*args)

        result = jax.lax.map(lambda args: one(*args), tuple(broadcast))
        if n_out == 1:
            return result, 0
        return tuple(result), tuple(0 for _ in range(n_out))

    def primal_batch(vals, dims):
        def bind_fn(p, ts):
            return prim.bind(p, ts)

        return _batch(vals, dims, bind_fn=bind_fn, n_out=2)

    batching.primitive_batchers[prim] = primal_batch

    def sens_batch(vals, dims):
        def bind_fn(p, ts, ckpt, dp):
            return sens_prim.bind(p, ts, ckpt, dp)

        return _batch(vals, dims, bind_fn=bind_fn, n_out=1)

    batching.primitive_batchers[sens_prim] = sens_batch

    return prim


class ODEProblem:
    """An ODE problem compiled from a Python RHS function.

    Wraps a user-supplied right-hand-side function, lowers it to a
    `DiffSL <https://martinjrobins.github.io/diffsl/>`_ source string, and
    compiles it for use with :func:`jax.jit`, :func:`jax.grad`,
    :func:`jax.jvp`, and :func:`jax.vmap`.

    The compiled solver is exposed via the :attr:`solve` method, which accepts
    ``params`` and a time span. Initial conditions are fixed at construction
    time from ``y0``.

    Supported AD operations
    -----------------------
    * ``jax.grad`` / ``jax.vjp`` — reverse-mode via diffsol discrete adjoint
    * ``jax.jvp`` / ``jax.jacfwd`` — forward-mode via diffsol sensitivities
    * ``jax.jit`` — compiled via XLA custom call
    * ``jax.vmap`` — sequential batching via ``lax.map``

    Not supported: higher-order derivatives (``grad(grad(...))``).
    Not supported: gradients w.r.t. ``y0`` (fixed at construction).

    Thread safety
    -------------
    An ``Arc<Mutex<SolveProblem>>`` serialises concurrent Rust-side solves.
    Concurrent Python-thread calls to :meth:`solve` will block but not deadlock.
    ``vmap`` sequential batching is safe.
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

        :param rhs: Right-hand-side function ``rhs(t, y, p) -> tuple[float, ...]``.
        :param y0: Initial state vector (shape determines ``n_state``). Fixed.
        :param params: Example parameter vector (shape determines ``n_params``).
        :param n_times: Number of output time points. Defaults to ``200``.
        """
        diffsl_src = make_diffsl_tuple(rhs, y0, params)
        solver = _rust.OdeSolver(diffsl_src)

        self.solver = solver
        self._handle = solver.handle()
        self.n_times = n_times
        self.n_state = len(y0)
        self.n_params = len(params)
        self._prims: dict = {}

    def solve(
        self,
        params: Float[Array, " params"],
        t_span: Float[Array, " 2"],
        ode_solver: ODE_SOLVER = "bdf",
    ) -> Tuple[Float[Array, " {n_times}"], Float[Array, "{n_times} state"]]:
        """Solve the ODE, returning ``(ts, ys)``.

        :param params: Parameter vector, shape ``(n_params,)``, dtype ``float64``.
        :param t_span: ``[t0, t_final]``, shape ``(2,)``, dtype ``float64``.
        :param ode_solver: Solver name — ``"bdf"``, ``"tsit45"``,
            ``"esdirk34"``, or ``"tr_bdf2"``.  Defaults to ``"bdf"``.
        :returns: ``(ts, ys)`` where ``ts`` has shape ``(n_times,)`` and
            ``ys`` has shape ``(n_times, n_state)``.
        """
        ode_solver_code = _method_code(ode_solver)
        if ode_solver_code not in self._prims:
            self._prims[ode_solver_code] = _make_primitive(
                self._handle,
                self.n_state,
                self.n_times,
                ode_solver_code,
                self.n_params,
            )
        prim = self._prims[ode_solver_code]

        params = jnp.asarray(params, dtype=jnp.float64)
        t_span = jnp.asarray(t_span, dtype=jnp.float64)

        ys, ts = prim.bind(params, t_span)
        return ts, ys
