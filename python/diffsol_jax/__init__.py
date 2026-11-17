import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import _lib, _lower, _params, _primitives
from ._lib import DiffsolJaxError
from ._lower import CallbackId, Equations
from ._params import System

__all__ = ["CallbackId", "DiffsolJaxError", "Model", "System"]

_ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
_LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
_MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


def _diagonal_mass(algebraic, nstates):
    """The diagonal mass of `algebraic`: an int meaning that last `algebraic` states are algebraic
    or a boolean mask of length `nstates` with True denoting an algebraic state."""
    if isinstance(algebraic, bool):
        raise DiffsolJaxError("algebraic must be an int or a boolean mask, not a bool")
    if isinstance(algebraic, (int, jnp.integer)):
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


class Model:
    """A diffsol solver for one problem: `Model(system, nstates, **options)`, or the functional
    form `Model(rhs, nstates, ...)` with `nparams` (flat parameters) or `variables_like` (pytree
    parameters). `solve(y0, params, t_eval)` returns `ys [..., nt, nout]`."""

    def __init__(
        self,
        rhs=None,
        nstates=None,
        nparams=None,
        out=None,
        nout=0,
        mass=None,
        algebraic=None,
        variables_like=None,
        ode_solver="bdf",
        linear_solver="default",
        matrix_type="nalgebra_dense",
        rtol=1e-6,
        atol=1e-6,
        ad="adjoint",
        max_checkpoints=8,
    ):
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError("diffsol_jax needs jax_enable_x64")
        if isinstance(rhs, _params.System):
            system = rhs
            given = {
                "nparams": nparams,
                "out": out,
                "mass": mass,
                "variables_like": variables_like,
            }
            extra = [k for k, v in given.items() if v is not None]
            extra += ["nout"] if nout else []
            if extra:
                raise DiffsolJaxError(
                    "Model(system, nstates) takes the equations from the System: "
                    f"do not also give {', '.join(extra)}"
                )
            if nstates is None:
                raise DiffsolJaxError(
                    "Model(system, nstates): give nstates, the length of the state that "
                    "System.__call__ returns"
                )
            rhs = lambda x, v, t: v(x, t)  # noqa: E731
            if type(system).out is not _params.System.out:
                out = lambda x, v, t: v.out(x, t)  # noqa: E731
                x0 = jax.ShapeDtypeStruct((nstates,), jnp.float64)
                t0 = jax.ShapeDtypeStruct((), jnp.float64)
                nout = jax.eval_shape(system.out, x0, t0).shape[0]
            if type(system).mass is not _params.System.mass:
                mass = lambda v, p, t: p.mass(v, t)  # noqa: E731
            variables_like = system
        elif rhs is None or nstates is None:
            raise DiffsolJaxError("give a System and nstates, or rhs and nstates")
        if (out is None) != (nout == 0):
            raise DiffsolJaxError("give out and nout > 0 together")
        if ad not in ("adjoint", "forward"):
            raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
        if (nparams is None) == (variables_like is None):
            raise DiffsolJaxError("give exactly one of nparams and variables_like")
        self.spec = None
        if variables_like is not None:
            self.spec = _params.ParamSpec(variables_like)
            nparams = self.spec.nparams
            rhs, out, mass = self.spec.bind_equations(rhs, out, mass)
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
        assert nparams is not None
        # diffsol's initial state is `init(p, t)`, a function of the parameters, so `y0` rides
        # in the tail of the internal parameter vector `[params, y0]`. Then `init` is a slice:
        # `init_sens` is the constant selection matrix `[0 | I]`, and the gradient with respect
        # to `y0` comes out of the existing sensitivity and adjoint callbacks with no special
        # case. The user equations only see the `params` head.
        rhs = _over_params(nparams, rhs)
        out = _over_params(nparams, out)
        mass = _over_params(nparams, mass)

        def init(p, t):
            return p[nparams:]

        self.eq = Equations(rhs, init, nstates, nparams + nstates, out, nout, mass)
        self.nstates, self.nparams, self.nout = nstates, nparams, nout
        self.nsolution = nout if nout > 0 else nstates
        self.rtol, self.atol, self.ad = float(rtol), float(atol), ad
        so = _lib.lib()
        self.id = _lib.call(
            so.diffsol_jax_model_new,
            nstates,
            self.eq.nparams,
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

    # ---- solve --------------------------------------------------------------------------------

    def _make_solve(self):
        kw = {"model": self, "rtol": self.rtol, "atol": self.atol}
        prim_cls = (
            _primitives.SolveAdjoint
            if self.ad == "adjoint"
            else _primitives.SolveForward
        )

        def check(y0, params, t_eval):
            t_eval = jnp.asarray(t_eval, jnp.float64)
            if t_eval.ndim != 1:
                raise DiffsolJaxError("t_eval must have one axis")
            y0 = jnp.asarray(y0, jnp.float64)
            if y0.ndim < 1 or y0.shape[-1] != self.nstates:
                raise DiffsolJaxError(f"y0 must have the shape [..., {self.nstates}]")
            if self.spec is not None:
                params = self.spec.ravel(params)
            else:
                params = jnp.asarray(params, jnp.float64)
                if params.ndim < 1 or params.shape[-1] != self.nparams:
                    raise DiffsolJaxError(
                        f"params must have the shape [..., {self.nparams}]"
                    )
            # the internal vector `[params, y0]`, with the leading axes broadcast together
            try:
                batch = jnp.broadcast_shapes(params.shape[:-1], y0.shape[:-1])
            except ValueError:
                raise DiffsolJaxError(
                    f"the leading axes of y0 {y0.shape[:-1]} and of params "
                    f"{params.shape[:-1]} do not broadcast together"
                ) from None
            params = jnp.broadcast_to(params, batch + (self.nparams,))
            y0 = jnp.broadcast_to(y0, batch + (self.nstates,))
            return jnp.concatenate([params, y0], axis=-1), t_eval

        def run(y0, params, t_eval) -> jax.Array:
            """Solve from the initial state `y0` (`[..., nstates]`) for `params`
            (`[..., nparams]`, or the variables pytree) at the times `t_eval [nt]`; returns
            `ys [..., nt, nout]`. Differentiable with respect to both `y0` and `params`."""
            params, t_eval = check(y0, params, t_eval)
            prim = prim_cls(jax.typeof(params), jax.typeof(t_eval), **kw)
            return prim(params, t_eval)

        return run
