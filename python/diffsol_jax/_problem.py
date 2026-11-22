from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from . import _equations
from ._layout import Layout
from ._native import Model
from ._types import (
    DTypeLike,
    DiffsolJaxError,
    LinearSolver,
    MatrixType,
    PyTree,
    Solver,
    solver_dtype,
)


@jax.tree_util.register_static
class ODEProblem:
    """An ODE (or semi-explicit index-1 DAE) and the diffsol solver for it.

    Everything the solver is built from is declared here; [`solve`][diffsol_jax.solve] then
    takes the initial state, the times and the parameter values. The solver, with its compiled
    callbacks and its adjoint checkpoints, is built once and lives as long as the problem.

    The state and the parameters are pytrees, declared by `state=` and `params=`. Each is
    either an int `n`, meaning one flat `dtype[n]` array, or a template pytree: its float-array
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
        dtype: the scalar type the solver works in, `float64` (the default) or `float32`. This
            is the dtype of `y0`, `params`, `ys` and the gradients, and of the `y`, `p` and `t`
            the equations see. It is a solver-precision choice, not a JAX x64 switch:
            `jax_enable_x64` is still required, and the times stay `float64` either way. An
            `f32` solve needs looser tolerances than the defaults -- around `1e-5` is the most
            `f32` can resolve.
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
        dtype: DTypeLike = jnp.float64,
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
            "dtype": dtype,
        }
        self.solver, self.linear_solver = solver, linear_solver
        self.matrix_type, self.max_checkpoints = matrix_type, max_checkpoints
        self.dtype = solver_dtype(dtype)

        self.state = Layout(state, "state", self.dtype, constants=False)
        self.params = Layout(params, "params", self.dtype)
        if self.state.n < 1:
            raise DiffsolJaxError("state must have at least one entry")
        self.nstates, self.nparams = self.state.n, self.params.n
        # what `ys` comes back in: the state, or whatever `out` returns
        self.solution = _equations.solution_layout(
            out, self.state, self.params, self.dtype
        )
        self.nout = 0 if out is None else self.solution.n
        self.nsolution = self.solution.n

        flat_mass = self._flat_mass(mass, algebraic)
        if flat_mass is not None and solver == "tsit45":
            raise DiffsolJaxError(
                "a problem with a mass matrix (a DAE) needs an implicit solver: "
                "solver='bdf', 'esdirk34' or 'tr_bdf2'"
            )
        self.eq = _equations.build(
            rhs, out, flat_mass, self.state, self.params, self.solution, self.dtype
        )
        self._model = Model(
            self.eq,
            solver=solver,
            linear_solver=linear_solver,
            matrix_type=matrix_type,
            max_checkpoints=max_checkpoints,
        )

    def _flat_mass(self, mass, algebraic):
        """The mass matrix action over the flat state, from `mass` or the `algebraic` mask."""
        if algebraic is not None:
            return _equations.diagonal_mass(self.state.mask(algebraic), self.dtype)
        flat_mass = _equations.over_flat(
            mass, "mass", self.state, self.params, self.state
        )
        if flat_mass is not None:
            _equations.check_mass_is_linear(flat_mass, self.nstates, self.nparams)
        return flat_mass

    def __repr__(self):
        parts = [repr(self.state), repr(self.params)]
        if self.solution is not self.state:
            parts.append(repr(self.solution))
        parts.append(f"solver={self.solver!r}")
        if self.dtype != jnp.float64:
            parts.append(f"dtype={self.dtype.name}")
        return f"ODEProblem({', '.join(parts)})"

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

    # ---- counters ----------------------------------------------------------------------------

    def clear_checkpoints(self):
        """Free the adjoint checkpoints kept for pending backward solves."""
        self._model.clear_checkpoints()

    def num_checkpoints(self) -> int:
        """How many adjoint checkpoints are alive."""
        return self._model.num_checkpoints()

    def num_recomputes(self) -> int:
        """The number of backward solves that ran the forward solve again, because the
        registry had freed their checkpoint."""
        return self._model.num_recomputes()

    def callback_stats(self, nbatch: int, reset: bool = False):
        """`(ncalls, nstaged, nviews)` for the solver with `nbatch` lanes: the number of callback
        calls, of inputs copied because they were not 64-byte aligned, and of PJRT views made
        (the misses of the view cache)."""
        return self._model.callback_stats(nbatch, reset)
