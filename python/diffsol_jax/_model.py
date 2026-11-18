"""The compiled side of a problem: the diffsol solver for one set of equations at one parameter
layout, and the JAX-lowered callbacks it calls."""

import dataclasses
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import _lib, _lower
from ._lib import DiffsolJaxError
from ._lower import Equations
from ._params import F64

ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


@dataclasses.dataclass(frozen=True)
class Config:
    """The options of a solve that the diffsol solver itself is built with, so two solves that
    differ in any of them use two solvers."""

    solver: str = "bdf"
    linear_solver: str = "default"
    matrix_type: str = "nalgebra_dense"
    max_checkpoints: int = 8

    def __post_init__(self):
        for name, table in (
            ("solver", ODE_SOLVERS),
            ("linear_solver", LINEAR_SOLVERS),
            ("matrix_type", MATRIX_TYPES),
        ):
            value = getattr(self, name)
            if value not in table:
                raise DiffsolJaxError(
                    f"{name} must be one of {', '.join(table)}, got {value!r}"
                )
        if self.max_checkpoints < 1:
            raise DiffsolJaxError("max_checkpoints must be at least 1")


def _diagonal_mass(algebraic, nstates):
    """The diagonal mass of `algebraic`: an int meaning that the last `algebraic` states are
    algebraic, or a boolean mask of length `nstates` with True denoting an algebraic state."""
    if isinstance(algebraic, int):
        if not 0 < algebraic < nstates:
            raise DiffsolJaxError(
                f"algebraic must be in [1, {nstates - 1}], got {algebraic}"
            )
        mask = [False] * (nstates - algebraic) + [True] * algebraic
    else:
        mask = list(algebraic)
        if len(mask) != nstates:
            raise DiffsolJaxError(f"the algebraic mask must have length {nstates}")
        if all(mask) or not any(mask):
            raise DiffsolJaxError("algebraic must mark some, not all, of the states")
    diff = jnp.array([0.0 if a else 1.0 for a in mask], F64)
    return lambda v, p, t: diff * v


def _check_mass_is_linear(mass, nstates, nparams):
    """Check at a random point that `mass(v, p, t)` is linear in `v` and independent of `p`"""
    rng = np.random.default_rng(0)
    v1, v2 = (
        jnp.asarray(rng.standard_normal(nstates)),
        jnp.asarray(rng.standard_normal(nstates)),
    )
    p1, p2 = (
        jnp.asarray(rng.standard_normal(nparams)),
        jnp.asarray(rng.standard_normal(nparams)),
    )
    t = 0.37
    a, b = 2.3, -1.7
    lhs = mass(a * v1 + b * v2, p1, t)
    rhs = a * mass(v1, p1, t) + b * mass(v2, p1, t)
    if not np.allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-9, atol=1e-9):
        raise DiffsolJaxError(
            "mass(v, p, t) must be linear in v (the action of a mass matrix)"
        )
    if not np.allclose(
        np.asarray(mass(v1, p1, t)), np.asarray(mass(v1, p2, t)), rtol=1e-9
    ):
        raise DiffsolJaxError(
            "mass(v, p, t) must not depend on p: the diffsol sensitivities have no dM/dp term, "
            "so the gradient would be wrong. Fold the parameter into rhs instead (for example "
            "divide the differential equations by it, or scale the algebraic equations)"
        )


def _over_params(nparams, f):
    """`f(a, p, t)` over the user parameters, as a function of the internal `[params, y0]`."""
    if f is None:
        return None
    return lambda a, p, t: f(a, p[:nparams], t)


def _spec(*shape):
    return jax.ShapeDtypeStruct(shape, F64)


class Model:
    """A diffsol solver, built for one `ODEProblem` at one state size, parameter layout and
    `Config`. Not constructed directly: `solve` builds (and caches) one as it needs it, and
    `ODEProblem.compile` returns the same object for its counters."""

    def __init__(self, problem=None, nstates=None, layout=None, config=None, **legacy):
        if legacy or layout is None or callable(problem):
            raise DiffsolJaxError(
                "Model is no longer the entry point: give the equations to "
                "ODEProblem(rhs, out=..., mass=..., algebraic=...) and solve them with "
                "solve(problem, y0, t_eval, args, solver=..., ad=...). nstates, nparams and "
                "nout are taken from y0 and args"
            )
        assert nstates is not None and config is not None
        self.problem, self.config, self.layout = problem, config, layout
        self.nstates, self.nparams = nstates, layout.nparams
        rhs, out, mass = layout.bind_equations(problem.rhs, problem.out, problem.mass)
        self.nout = 0
        if out is not None:
            shape = jax.eval_shape(
                out, _spec(nstates), _spec(self.nparams), _spec()
            ).shape
            if len(shape) != 1:
                raise DiffsolJaxError(f"out must return [nout], got the shape {shape}")
            self.nout = int(shape[0])
        if problem.algebraic is not None:
            mass = _diagonal_mass(problem.algebraic, nstates)
        elif mass is not None:
            _check_mass_is_linear(mass, nstates, self.nparams)
        if mass is not None and config.solver == "tsit45":
            raise DiffsolJaxError(
                "a problem with a mass matrix (a DAE) needs an implicit solver: "
                "solver='bdf', 'esdirk34' or 'tr_bdf2'"
            )
        # diffsol's initial state is `init(p, t)`, a function of the parameters, so `y0` rides
        # in the tail of the internal parameter vector `[params, y0]`. Then `init` is a slice:
        # `init_sens` is the constant selection matrix `[0 | I]`, and the gradient with respect
        # to `y0` comes out of the existing sensitivity and adjoint callbacks with no special
        # case. The user equations only see the `params` head.
        nparams = self.nparams
        self.eq = Equations(
            _over_params(nparams, rhs),
            lambda p, t: p[nparams:],
            nstates,
            nparams + nstates,
            _over_params(nparams, out),
            self.nout,
            _over_params(nparams, mass),
        )
        self.nsolution = self.nout if self.nout > 0 else nstates
        so = _lib.lib()
        self.id = _lib.call(
            so.diffsol_jax_model_new,
            nstates,
            self.eq.nparams,
            self.nout,
            MATRIX_TYPES[config.matrix_type],
            LINEAR_SOLVERS[config.linear_solver],
            ODE_SOLVERS[config.solver],
            config.max_checkpoints,
        )
        self._finalizer = weakref.finalize(self, so.diffsol_jax_model_free, self.id)
        self._compile_options = None

    def __repr__(self):
        return (
            f"Model(nstates={self.nstates}, nparams={self.nparams}, nout={self.nout}, "
            f"solver={self.config.solver!r})"
        )

    # ---- the solve arguments ------------------------------------------------------------------

    def pack(self, y0, args) -> jax.Array:
        """The internal parameter vector `[params, y0]`, with the leading axes of the parameters
        and of `y0` broadcast together into the lanes of one batched solve."""
        y0 = jnp.asarray(y0, F64)
        if y0.ndim < 1 or y0.shape[-1] != self.nstates:
            raise DiffsolJaxError(f"y0 must have the shape [..., {self.nstates}]")
        params = self.layout.ravel(args)
        if params.ndim < 1 or params.shape[-1] != self.nparams:
            raise DiffsolJaxError(f"args must have the shape [..., {self.nparams}]")
        try:
            batch = jnp.broadcast_shapes(params.shape[:-1], y0.shape[:-1])
        except ValueError:
            raise DiffsolJaxError(
                f"the leading axes of y0 {y0.shape[:-1]} and of args "
                f"{params.shape[:-1]} do not broadcast together"
            ) from None
        params = jnp.broadcast_to(params, batch + (self.nparams,))
        y0 = jnp.broadcast_to(y0, batch + (self.nstates,))
        return jnp.concatenate([params, y0], axis=-1)

    # ---- callbacks ----------------------------------------------------------------------------

    def _options(self):
        if self._compile_options is None:
            from jax._src.lib import xla_client

            self._compile_options = xla_client.CompileOptions().SerializeAsString()
        return self._compile_options

    def _add(self, cb, nbatch, p):
        so = _lib.lib()
        if so.diffsol_jax_model_has_variant(self.id, nbatch, int(cb), p):
            return
        code = _lower.lower(self.eq, cb, nbatch, p)
        opts = self._options()
        _lib.call(
            so.diffsol_jax_model_add_variant,
            self.id,
            nbatch,
            int(cb),
            p,
            code,
            len(code),
            opts,
            len(opts),
        )

    def ensure_callbacks(self, kind, nbatch, ndirections=1):
        """Compile the callbacks that a solve of `kind` with `nbatch` lanes needs (called by the
        lowering rules). All the callbacks of the model get a `P = 1` executable, so the set of
        callbacks of the diffsol ODE does not change between solve kinds."""
        available = self.eq.available()
        for cb in available:
            self._add(cb, nbatch, 1)
        if kind == "fwd_sens" and self.eq.nparams > 1:
            for cb in _lower.FWD_SENS_MULTI_P:
                if cb in available:
                    self._add(cb, nbatch, self.eq.nparams)
        if kind == "vjp_bkwd" and ndirections > 1:
            for cb in _lower.VJP_BKWD_MULTI_P:
                if cb in available:
                    self._add(cb, nbatch, ndirections)

    def clear_checkpoints(self):
        _lib.lib().diffsol_jax_model_clear_checkpoints(self.id)

    def num_checkpoints(self):
        return _lib.lib().diffsol_jax_model_num_checkpoints(self.id)

    def callback_stats(self, nbatch, reset=False):
        """`(ncalls, nstaged, nviews)` for the solver with `nbatch` lanes: the number of callback
        calls, of inputs copied because they were not 64-byte aligned, and of PJRT views made
        (the misses of the view cache)."""
        import ctypes

        ncalls, nstaged, nviews = (
            ctypes.c_uint64(),
            ctypes.c_uint64(),
            ctypes.c_uint64(),
        )
        ok = _lib.lib().diffsol_jax_model_stats(
            self.id,
            nbatch,
            ctypes.byref(ncalls),
            ctypes.byref(nstaged),
            ctypes.byref(nviews),
            int(reset),
        )
        return (ncalls.value, nstaged.value, nviews.value) if ok else (0, 0, 0)

    def num_recomputes(self):
        """The number of backward solves that ran the forward solve again, because the
        registry had freed their checkpoint."""
        return _lib.lib().diffsol_jax_model_num_recomputes(self.id)
