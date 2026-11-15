"""diffsol solves as JAX operations, with the ODE equations given as JAX functions.

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from diffsol_jax import Model

    model = Model(
        rhs=lambda x, p, t: -p[0] * x,
        init=lambda p, t: jnp.ones(2),
        nstates=2, nparams=1,
    )
    ys = model.solve(params, t_eval)       # params [..., nparams] -> ys [..., nt, nout]
    jax.grad(lambda p: model.solve(p, t_eval).sum())(params)

The functions act on one lane. diffsol calls the callbacks (compiled with the PJRT C API) for
all the lanes of a solve together. `jax.vmap` over `params` gives more lanes in one solve.

The solve is a `jax.experimental.hijax.HiPrim` (see `_primitives`). Forward mode (`jax.jvp`,
`jax.jacfwd`) uses diffsol forward sensitivities on every model. Reverse mode (`jax.grad`,
`jax.jacrev`) depends on `ad`: with `ad="adjoint"` (the default) it uses the diffsol adjoint
(the forward solve keeps a checkpoint in C++ and gives an int64 handle to the backward solve;
the model keeps at most `max_checkpoints` checkpoints, and a backward solve whose checkpoint
was freed runs the forward solve again from the `params` and `t_eval` that it also gets, so
`max_checkpoints` controls only the memory); with `ad="forward"` it is the transpose of the
forward sensitivities.

A model with a mass matrix solves `M(t) u' = f(u, p, t)`: `mass(v, p, t)` is the action `M v`,
linear in `v` and independent of `p` (the diffsol sensitivities have no dM/dp term, so a
parameter-dependent mass is rejected). A singular `M` gives a semi-explicit index-1 DAE;
`algebraic=` (an int or a boolean mask) is a shortcut for the diagonal mass with zeros on the
algebraic states. The
initial state is projected onto the algebraic constraints (diffsol solves the algebraic
variables at `t0`), the gradients account for it, and a DAE needs an implicit solver
(`bdf`, `esdirk34` or `tr_bdf2`). Higher-index DAEs must be reduced to index 1 first.

There is no derivative with respect to `t_eval`.
"""

import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import _lib, _lower, _primitives
from ._lib import DiffsolJaxError
from ._lower import CallbackId, Equations

__all__ = ["CallbackId", "DiffsolJaxError", "Model"]

_ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
_LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
_MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


def _diagonal_mass(algebraic, nstates):
    """The diagonal mass of `algebraic`: an int (the last `algebraic` states are algebraic) or a
    boolean mask of length `nstates` (True marks an algebraic state)."""
    if isinstance(algebraic, (int, jnp.integer)) and not isinstance(algebraic, bool):
        if not 0 < algebraic < nstates:
            raise DiffsolJaxError(
                f"algebraic must be in [1, {nstates - 1}], got {algebraic}"
            )
        mask = [False] * (nstates - algebraic) + [True] * algebraic
    else:
        mask = [bool(a) for a in algebraic]
        if len(mask) != nstates:
            raise DiffsolJaxError(f"the algebraic mask must have length {nstates}")
        if all(mask) or not any(mask):
            raise DiffsolJaxError("algebraic must mark some, not all, of the states")
    diff = jnp.array([0.0 if a else 1.0 for a in mask], jnp.float64)
    return lambda v, p, t: diff * v


def _check_mass_is_linear(mass, nstates, nparams):
    """`mass(v, p, t)` must be the action of a matrix on `v`: check linearity in `v`, and that
    the matrix does not depend on `p`, at a random point, once, at model construction. The
    sensitivity and adjoint equations of diffsol have no dM/dp term, so the gradient of a
    parameter-dependent mass would silently miss it."""
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


class Model:
    def __init__(
        self,
        rhs,
        init,
        nstates,
        nparams,
        out=None,
        nout=0,
        mass=None,
        algebraic=None,
        ode_solver="bdf",
        linear_solver="default",
        matrix_type="nalgebra_dense",
        rtol=1e-6,
        atol=1e-6,
        ad="adjoint",
        max_checkpoints=8,
    ):
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError(
                "diffsol_jax needs jax_enable_x64 (the solver uses f64)"
            )
        if (out is None) != (nout == 0):
            raise DiffsolJaxError("give out and nout > 0 together")
        if ad not in ("adjoint", "forward"):
            raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
        if algebraic is not None:
            if mass is not None:
                raise DiffsolJaxError("give mass or algebraic, not both")
            mass = _diagonal_mass(algebraic, nstates)
        elif mass is not None:
            _check_mass_is_linear(mass, nstates, nparams)
        if mass is not None and ode_solver == "tsit45":
            raise DiffsolJaxError(
                "a model with a mass matrix (a DAE) needs an implicit solver: "
                "ode_solver='bdf', 'esdirk34' or 'tr_bdf2'"
            )
        self.eq = Equations(rhs, init, nstates, nparams, out, nout, mass)
        self.nstates, self.nparams, self.nout = nstates, nparams, nout
        self.nsolution = nout if nout > 0 else nstates
        self.rtol, self.atol, self.ad = float(rtol), float(atol), ad
        so = _lib.lib()
        self.id = _lib.call(
            so.diffsol_jax_model_new,
            nstates,
            nparams,
            nout,
            _MATRIX_TYPES[matrix_type],
            _LINEAR_SOLVERS[linear_solver],
            _ODE_SOLVERS[ode_solver],
            max_checkpoints,
        )
        self._finalizer = weakref.finalize(self, so.diffsol_jax_model_free, self.id)
        self._compile_options = None
        self.solve = self._make_solve()

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
        if kind == "fwd_sens" and self.nparams > 1:
            for cb in _lower.FWD_SENS_MULTI_P:
                if cb in available:
                    self._add(cb, nbatch, self.nparams)
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

    # ---- solve --------------------------------------------------------------------------------

    def _make_solve(self):
        kw = {"model": self, "rtol": self.rtol, "atol": self.atol}
        prim_cls = (
            _primitives.SolveAdjoint
            if self.ad == "adjoint"
            else _primitives.SolveForward
        )

        def check(params, t_eval):
            params = jnp.asarray(params, jnp.float64)
            t_eval = jnp.asarray(t_eval, jnp.float64)
            if params.ndim < 1 or params.shape[-1] != self.nparams:
                raise DiffsolJaxError(
                    f"params must have the shape [..., {self.nparams}]"
                )
            if t_eval.ndim != 1:
                raise DiffsolJaxError("t_eval must have one axis")
            return params, t_eval

        def run(params, t_eval):
            """Solve for each lane of `params [..., nparams]` at the times `t_eval [nt]`.
            Return `ys [..., nt, nout]` (`nout = nstates` if the model has no `out`)."""
            params, t_eval = check(params, t_eval)
            prim = prim_cls(jax.typeof(params), jax.typeof(t_eval), **kw)
            return prim(params, t_eval)

        return run
