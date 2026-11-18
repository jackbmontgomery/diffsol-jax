import weakref
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import _lib, _lower
from ._lib import DiffsolJaxError
from ._lower import Equations
from ._params import F64, Layout
from ._types import LinearSolver, MatrixType, PyTree, Solver

ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


def _one_of(name, value, table):
    if value not in table:
        raise DiffsolJaxError(
            f"{name} must be one of {', '.join(table)}, got {value!r}"
        )
    return table[value]


def _diagonal_mass(mask):
    """The diagonal mass of a boolean mask over the flat state, True denoting an algebraic
    entry: the rows of `rhs` it selects become algebraic equations."""
    if all(mask) or not any(mask):
        raise DiffsolJaxError("algebraic must mark some, not all, of the state")
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


def _spec(*shape):
    return jax.ShapeDtypeStruct(shape, F64)


@jax.tree_util.register_static
class ODEProblem:
    """An ODE (or semi-explicit index-1 DAE) and the diffsol solver for it.

    Everything the solver is built from is declared here; [`solve`][diffsol_jax.solve] then
    takes the initial state, the times and the parameter values. The solver, with its compiled
    callbacks and its adjoint checkpoints, is built once and lives as long as the problem.

    The state and the parameters are pytrees, declared by `state=` and `params=`. Each is
    either an int `n`, meaning one flat `f64[n]` array, or a template pytree: its float-array
    leaves carry the shapes and dtypes, and `jax.eval_shape(make_net, key)` does as well as a
    network. The equations then act on one lane of those pytrees.

    Args:
        rhs: `rhs(y, p, t) -> y-like`, the right-hand side: `y` and the result are in the
            `state` structure, `p` is in the `params` structure, and `t` is a scalar.
        state: the layout of the state, an int or a template. Every leaf must be a float array.
        params: the layout of the parameters, an int, `None` for no parameters, or a template
            whose float-array leaves are the parameters and whose other leaves the equations
            capture as constants (their values coming from the template).
        out: `out(y, p, t) -> pytree`, an output to return in place of the state. Its structure
            is inferred by shape tracing, and `ys` comes back in it.
        mass: `mass(v, p, t) -> v-like`, the mass matrix action `M(t) v`, with `v` in the state
            structure. Must be linear in `v` and must not use `p`.
        algebraic: a `mass` shortcut for a semi-explicit DAE, marking the rows of `rhs` that
            are algebraic equations: an int (the last `algebraic` entries of the flat state), a
            sequence of `nstates` bools, or a pytree of bools matching `state`.
        solver: the diffsol solver: `bdf`, `esdirk34`, `tr_bdf2`, or `tsit45` (explicit, so not
            for a DAE).
        linear_solver: `default`, `lu`, or `klu`.
        matrix_type: `nalgebra_dense`, `faer_dense`, or `faer_sparse`.
        max_checkpoints: how many adjoint checkpoints to keep alive at once.
    """

    def __init__(
        self,
        rhs: Callable,
        state: int | PyTree,
        params: int | PyTree = None,
        *,
        out: Callable | None = None,
        mass: Callable | None = None,
        algebraic: int | Sequence[bool] | PyTree | None = None,
        solver: Solver = "bdf",
        linear_solver: LinearSolver = "default",
        matrix_type: MatrixType = "nalgebra_dense",
        max_checkpoints: int = 8,
    ):
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError("diffsol_jax needs jax_enable_x64")
        if not callable(rhs):
            raise DiffsolJaxError("rhs must be a function rhs(y, p, t) -> y-like")
        if mass is not None and algebraic is not None:
            raise DiffsolJaxError("give mass or algebraic, not both")
        if max_checkpoints < 1:
            raise DiffsolJaxError("max_checkpoints must be at least 1")
        # the arguments as given, so that `replace` can rebuild the problem
        self._declared = {
            "rhs": rhs,
            "state": state,
            "params": params,
            "out": out,
            "mass": mass,
            "algebraic": algebraic,
            "solver": solver,
            "linear_solver": linear_solver,
            "matrix_type": matrix_type,
            "max_checkpoints": max_checkpoints,
        }
        self.solver, self.linear_solver = solver, linear_solver
        self.matrix_type, self.max_checkpoints = matrix_type, max_checkpoints
        self.state = Layout(state, "state", constants=False)
        self.params = Layout(params, "params")
        if self.state.n < 1:
            raise DiffsolJaxError("state must have at least one entry")
        nstates, nparams = self.state.n, self.params.n
        self.nstates, self.nparams = nstates, nparams

        # the wrappers close over the layouts, not over `self`: a reference from `self.eq` back
        # to `self` would be a cycle, and the native solver would outlive the problem until the
        # next gc pass
        state, params_layout = self.state, self.params

        def over_flat(f, name, result):
            """`f` over the pytrees, as a function of the flat vectors."""
            if f is None:
                return None

            def wrapped(a, p, t):
                y = f(state.unravel(a), params_layout.unravel(p), t)
                return result.ravel(y, f"the result of {name}")

            return wrapped

        # what `ys` comes back in: the state, or whatever `out` returns
        self.solution = self.state
        if out is not None:
            self.solution = Layout(
                jax.eval_shape(
                    lambda a, p, t: out(state.unravel(a), params_layout.unravel(p), t),
                    _spec(nstates),
                    _spec(nparams),
                    _spec(),
                ),
                "out",
                constants=False,
            )
        self.nout = 0 if out is None else self.solution.n
        self.nsolution = self.solution.n
        if algebraic is not None:
            mask = self.state.mask(algebraic)
            flat_mass = _diagonal_mass(mask)
        else:
            flat_mass = over_flat(mass, "mass", self.state)
            if flat_mass is not None:
                _check_mass_is_linear(flat_mass, nstates, nparams)
        if flat_mass is not None and solver == "tsit45":
            raise DiffsolJaxError(
                "a problem with a mass matrix (a DAE) needs an implicit solver: "
                "solver='bdf', 'esdirk34' or 'tr_bdf2'"
            )

        # diffsol's initial state is `init(p, t)`, a function of the parameters, so `y0` rides
        # in the tail of the internal parameter vector `[params, y0]`. Then `init` is a slice:
        # `init_sens` is the constant selection matrix `[0 | I]`, and the gradient with respect
        # to `y0` comes out of the existing sensitivity and adjoint callbacks with no special
        # case. The user equations only see the `params` head.
        def over_params(f):
            return None if f is None else lambda a, p, t: f(a, p[:nparams], t)

        self.eq = Equations(
            over_params(over_flat(rhs, "rhs", self.state)),
            lambda p, t: p[nparams:],
            nstates,
            nparams + nstates,
            over_params(over_flat(out, "out", self.solution)),
            self.nout,
            over_params(flat_mass),
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
        parts = [repr(self.state), repr(self.params)]
        if self.solution is not self.state:
            parts.append(repr(self.solution))
        return f"ODEProblem({', '.join(parts)}, solver={self.solver!r})"

    def replace(self, **changes) -> "ODEProblem":
        """A problem with the given arguments replaced, and so with its own diffsol solver:
        `problem.replace(solver="tsit45")`."""
        unknown = set(changes) - set(self._declared)
        if unknown:
            raise DiffsolJaxError(f"replace got unknown arguments: {sorted(unknown)}")
        return ODEProblem(**{**self._declared, **changes})

    # ---- the solve arguments ------------------------------------------------------------------

    def _pack(self, y0, params) -> jax.Array:
        """The internal parameter vector `[params, y0]` of one lane. A `jax.vmap` over either
        batches it through `concatenate` and into the lanes of one batched solve."""
        hint = "batch with jax.vmap rather than by adding axes to the leaves"
        return jnp.concatenate(
            [
                self.params.ravel(params, hint=hint),
                self.state.ravel(y0, "y0", hint=hint),
            ]
        )

    def _unpack(self, ys: jax.Array):
        """`ys [..., nt, nsolution]` as the solution pytree, each leaf `[..., nt, *shape]`."""
        return self.solution.unpack(ys)

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
