"""The problem: the equations, the sizes, and the diffsol solver they are built into."""

import weakref
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import _lib, _lower, _params
from ._lib import DiffsolJaxError
from ._lower import Equations
from ._params import F64
from ._types import Args, LinearSolver, MatrixType, Solver

ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


def _one_of(name, value, table):
    if value not in table:
        raise DiffsolJaxError(
            f"{name} must be one of {', '.join(table)}, got {value!r}"
        )
    return table[value]


def _normalise_algebraic(algebraic):
    if algebraic is None:
        return None
    if isinstance(algebraic, bool):
        raise DiffsolJaxError("algebraic must be an int or a boolean mask, not a bool")
    if isinstance(algebraic, (int, np.integer)):
        return int(algebraic)
    if isinstance(algebraic, (Sequence, jax.Array, np.ndarray)):
        return tuple(bool(a) for a in algebraic)
    raise DiffsolJaxError(
        f"algebraic must be an int or a boolean mask, got {type(algebraic).__name__}"
    )


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


# a problem holds functions, sizes and a native solver, never JAX arrays: as a static pytree it
# can be an argument of a jitted or vmapped function (`jax.jit(solve)`, `in_axes=None`) without
# being traced.
@jax.tree_util.register_static
class ODEProblem:
    """An ODE (or semi-explicit index-1 DAE) and the diffsol solver for it.

    Everything the solver is built from is declared here; [`solve`][diffsol_jax.solve] then
    takes the initial state, the times and the parameter values. The solver, with its compiled
    callbacks and its adjoint checkpoints, is built once and lives as long as the problem.

    Every equation takes one lane: the state `x [nstates]` (or a vector `v` of its size), the
    parameters `p`, and the scalar time `t`.

    Args:
        rhs: `rhs(x, p, t) -> [nstates]`, the right-hand side.
        nstates: the length of the state.
        params: the parameter layout. An int is a flat vector `p [nparams]`, whose leading axes
            in a solve are lanes of a batched solve; `None` is no parameters; anything else is a
            template pytree (an equinox module, a dataclass, a dict) whose float-array leaves
            are the parameters and whose other leaves the equations capture as constants. The
            `args` of a solve then supply the values, and `jax.grad` returns the same structure.
            Only shapes and dtypes are read from the parameter leaves, so
            `jax.eval_shape(make_net, key)` works as well as a network.
        out: `out(x, p, t) -> [nout]`, an output to return in place of the state. `nout` is
            inferred by shape tracing.
        mass: `mass(v, p, t) -> [nstates]`, the mass matrix action `M(t) v`. Must be linear in
            `v` and must not use `p`.
        algebraic: a `mass` shortcut for a semi-explicit DAE: an int, meaning that the last
            `algebraic` rows of `rhs` are algebraic equations, or a boolean mask over the
            states.
        solver: the diffsol solver: `bdf`, `esdirk34`, `tr_bdf2`, or `tsit45` (explicit, so not
            for a DAE).
        linear_solver: `default`, `lu`, or `klu`.
        matrix_type: `nalgebra_dense`, `faer_dense`, or `faer_sparse`.
        max_checkpoints: how many adjoint checkpoints to keep alive at once.
    """

    def __init__(
        self,
        rhs: Callable,
        nstates: int,
        params: Args = None,
        *,
        out: Callable | None = None,
        mass: Callable | None = None,
        algebraic: int | Sequence[bool] | None = None,
        solver: Solver = "bdf",
        linear_solver: LinearSolver = "default",
        matrix_type: MatrixType = "nalgebra_dense",
        max_checkpoints: int = 8,
    ):
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError("diffsol_jax needs jax_enable_x64")
        if not callable(rhs):
            raise DiffsolJaxError("rhs must be a function rhs(x, p, t) -> [nstates]")
        if isinstance(nstates, bool) or not isinstance(nstates, (int, np.integer)):
            raise DiffsolJaxError("nstates must be an int, the length of the state")
        if nstates < 1:
            raise DiffsolJaxError(f"nstates must be at least 1, got {nstates}")
        if mass is not None and algebraic is not None:
            raise DiffsolJaxError("give mass or algebraic, not both")
        if max_checkpoints < 1:
            raise DiffsolJaxError("max_checkpoints must be at least 1")
        algebraic = _normalise_algebraic(algebraic)
        # the arguments as given, so that `replace` can rebuild the problem
        self._declared = {
            "rhs": rhs,
            "nstates": int(nstates),
            "params": params,
            "out": out,
            "mass": mass,
            "algebraic": algebraic,
            "solver": solver,
            "linear_solver": linear_solver,
            "matrix_type": matrix_type,
            "max_checkpoints": max_checkpoints,
        }
        self.rhs, self.out, self.mass, self.algebraic = rhs, out, mass, algebraic
        self.solver, self.linear_solver = solver, linear_solver
        self.matrix_type, self.max_checkpoints = matrix_type, max_checkpoints
        self.nstates = int(nstates)
        self.layout = _params.layout(params)
        self.nparams = self.layout.nparams
        nparams = self.nparams

        rhs, out, mass = self.layout.bind_equations(rhs, out, mass)
        self.nout = 0
        if out is not None:
            shape = jax.eval_shape(out, _spec(nstates), _spec(nparams), _spec()).shape
            if len(shape) != 1:
                raise DiffsolJaxError(f"out must return [nout], got the shape {shape}")
            self.nout = int(shape[0])
        self.nsolution = self.nout if self.nout > 0 else self.nstates
        if algebraic is not None:
            mass = _diagonal_mass(algebraic, nstates)
        elif mass is not None:
            _check_mass_is_linear(mass, nstates, nparams)
        if mass is not None and solver == "tsit45":
            raise DiffsolJaxError(
                "a problem with a mass matrix (a DAE) needs an implicit solver: "
                "solver='bdf', 'esdirk34' or 'tr_bdf2'"
            )
        # diffsol's initial state is `init(p, t)`, a function of the parameters, so `y0` rides
        # in the tail of the internal parameter vector `[params, y0]`. Then `init` is a slice:
        # `init_sens` is the constant selection matrix `[0 | I]`, and the gradient with respect
        # to `y0` comes out of the existing sensitivity and adjoint callbacks with no special
        # case. The user equations only see the `params` head.
        self.eq = Equations(
            _over_params(nparams, rhs),
            lambda p, t: p[nparams:],
            nstates,
            nparams + nstates,
            _over_params(nparams, out),
            self.nout,
            _over_params(nparams, mass),
        )
        so = _lib.lib()
        self._id = _lib.call(
            so.diffsol_jax_model_new,
            nstates,
            self.eq.nparams,
            self.nout,
            _one_of("matrix_type", matrix_type, MATRIX_TYPES),
            _one_of("linear_solver", linear_solver, LINEAR_SOLVERS),
            _one_of("solver", solver, ODE_SOLVERS),
            max_checkpoints,
        )
        # the solver is freed with the problem
        self._finalizer = weakref.finalize(self, so.diffsol_jax_model_free, self._id)
        self._compile_options = None

    def __repr__(self):
        return (
            f"ODEProblem(nstates={self.nstates}, {self.layout!r}, nout={self.nout}, "
            f"solver={self.solver!r})"
        )

    def replace(self, **changes) -> "ODEProblem":
        """A problem with the given arguments replaced, and so with its own diffsol solver:
        `problem.replace(solver="tsit45")`."""
        unknown = set(changes) - set(self._declared)
        if unknown:
            raise DiffsolJaxError(f"replace got unknown arguments: {sorted(unknown)}")
        return ODEProblem(**{**self._declared, **changes})

    # ---- the solve arguments ------------------------------------------------------------------

    def _pack(self, y0, args) -> jax.Array:
        """The internal parameter vector `[params, y0]`, with the leading axes of the parameters
        and of `y0` broadcast together into the lanes of one batched solve."""
        y0 = jnp.asarray(y0, F64)
        if y0.ndim < 1 or y0.shape[-1] != self.nstates:
            raise DiffsolJaxError(
                f"y0 must have the shape [..., {self.nstates}], got {y0.shape}"
            )
        params = self.layout.ravel(args)
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
        if so.diffsol_jax_model_has_variant(self._id, nbatch, int(cb), p):
            return
        code = _lower.lower(self.eq, cb, nbatch, p)
        opts = self._options()
        _lib.call(
            so.diffsol_jax_model_add_variant,
            self._id,
            nbatch,
            int(cb),
            p,
            code,
            len(code),
            opts,
            len(opts),
        )

    def _ensure_callbacks(self, kind, nbatch, ndirections=1):
        """Compile the callbacks that a solve of `kind` with `nbatch` lanes needs (called by the
        lowering rules). All the callbacks get a `P = 1` executable, so the set of callbacks of
        the diffsol ODE does not change between solve kinds."""
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

    # ---- counters ----------------------------------------------------------------------------

    def clear_checkpoints(self):
        """Free the adjoint checkpoints kept for pending backward solves."""
        _lib.lib().diffsol_jax_model_clear_checkpoints(self._id)

    def num_checkpoints(self) -> int:
        """How many adjoint checkpoints are alive."""
        return _lib.lib().diffsol_jax_model_num_checkpoints(self._id)

    def num_recomputes(self) -> int:
        """The number of backward solves that ran the forward solve again, because the
        registry had freed their checkpoint."""
        return _lib.lib().diffsol_jax_model_num_recomputes(self._id)

    def callback_stats(self, nbatch: int, reset: bool = False):
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
            self._id,
            nbatch,
            ctypes.byref(ncalls),
            ctypes.byref(nstaged),
            ctypes.byref(nviews),
            int(reset),
        )
        return (ncalls.value, nstaged.value, nviews.value) if ok else (0, 0, 0)
