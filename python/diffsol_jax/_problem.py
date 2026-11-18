"""The problem: the equations, and nothing else."""

import dataclasses
import warnings
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import _params
from ._lib import DiffsolJaxError
from ._model import Config, Model
from ._types import Args, ArrayLike, LinearSolver, MatrixType, Solver

# the solvers built so far, keyed by everything they are built from. A solve looks its solver up
# here, so repeated solves of the same problem at the same shapes share one solver (and so one
# set of compiled callbacks and one checkpoint registry).
_MODELS: dict[object, Model] = {}
# a solver holds native memory, so warn once if they are piling up: the usual cause is a problem
# built inside the solve loop, so that every solve has a new `rhs` and misses the cache
_WARN_AT = 64
_warned = False


def clear_cache():
    """Drop the solvers built so far. The next solve builds (and compiles) them again."""
    global _warned
    _MODELS.clear()
    _warned = False


def _warn_if_many():
    global _warned
    if len(_MODELS) >= _WARN_AT and not _warned:
        _warned = True
        warnings.warn(
            f"diffsol_jax has built {len(_MODELS)} solvers. Each one holds native memory, and "
            "they are kept until clear_cache(). A problem built inside the solve loop misses "
            "the cache every time (its rhs is a new function): build the ODEProblem once, "
            "outside",
            stacklevel=3,
        )


def _normalise_algebraic(algebraic):
    if algebraic is None:
        return None
    if isinstance(algebraic, bool):
        raise DiffsolJaxError("algebraic must be an int or a boolean mask, not a bool")
    if isinstance(algebraic, (int, jnp.integer)):
        return int(algebraic)
    if isinstance(algebraic, (Sequence, jax.Array, np.ndarray)):
        return tuple(bool(a) for a in algebraic)
    raise DiffsolJaxError(
        f"algebraic must be an int or a boolean mask, got {type(algebraic).__name__}"
    )


# a problem holds functions and sizes, never arrays: as a static pytree it can be an argument of
# a jitted or vmapped function (`jax.jit(solve)`, `in_axes=None`) without being traced.
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class ODEProblem:
    """The equations of an ODE (or of a semi-explicit index-1 DAE).

    Every equation takes one lane: the state `x [nstates]` (or a vector `v` of its size), the
    parameters `p`, and the scalar time `t`. The parameters are whatever is passed to
    [`solve`][diffsol_jax.solve] as `args`: one array `[nparams]`, or a pytree whose float-array
    leaves are the parameters (an equinox module, a dataclass, a dict) and whose other leaves
    the equations capture as constants.

    Args:
        rhs: `rhs(x, p, t) -> [nstates]`, the right-hand side.
        out: `out(x, p, t) -> [nout]`, an output to return in place of the state.
        mass: `mass(v, p, t) -> [nstates]`, the mass matrix action `M(t) v`. Must be linear
            in `v` and must not use `p`.
        algebraic: a `mass` shortcut for a semi-explicit DAE: an int, meaning that the last
            `algebraic` rows of `rhs` are algebraic equations, or a boolean mask over the states.
    """

    rhs: Callable
    out: Callable | None = None
    mass: Callable | None = None
    algebraic: int | tuple[bool, ...] | None = None

    def __post_init__(self):
        if not callable(self.rhs):
            raise DiffsolJaxError("rhs must be a function rhs(x, p, t) -> [nstates]")
        if self.mass is not None and self.algebraic is not None:
            raise DiffsolJaxError("give mass or algebraic, not both")
        object.__setattr__(self, "algebraic", _normalise_algebraic(self.algebraic))

    def compile(
        self,
        y0: ArrayLike,
        args: Args = None,
        *,
        solver: Solver = "bdf",
        linear_solver: LinearSolver = "default",
        matrix_type: MatrixType = "nalgebra_dense",
        max_checkpoints: int = 8,
    ) -> Model:
        """The diffsol solver for this problem at the shapes of `y0` and `args`, built on the
        first call and cached. [`solve`][diffsol_jax.solve] calls this with its own options, so
        the returned solver is the one it uses: call it for the counters of
        [`Model`][diffsol_jax.Model] (the callback statistics, the checkpoints), or to pay the
        construction cost before the first solve."""
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError("diffsol_jax needs jax_enable_x64")
        y0 = jnp.asarray(y0, _params.F64)
        if y0.ndim < 1:
            raise DiffsolJaxError("y0 must have the shape [..., nstates]")
        nstates = int(y0.shape[-1])
        layout = _params.layout(args)
        config = Config(solver, linear_solver, matrix_type, max_checkpoints)
        key = (self, nstates, layout.key, config)
        model = _MODELS.get(key)
        if model is None:
            model = _MODELS[key] = Model(self, nstates, layout, config)
            _warn_if_many()
        return model
