import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.hijax import (
    HiPrim,
    ShapedArray,
    Zero,
    bdims_at_front,
    vjp_from_jvp,
)


def _ys_aval(params_aval, t_eval_aval, problem):
    """`ys [*B, nt, nsolution]`, in the dtype of `params`: that is the model's dtype, since
    `ODEProblem._pack` ravels into it."""
    return ShapedArray(
        params_aval.shape[:-1] + (t_eval_aval.shape[0], problem.nsolution),
        params_aval.dtype,
    )


class _DiffsolPrim(HiPrim):
    """A primitive that calls diffsol through the FFI. It carries the problem and the solver
    settings, which together are what an FFI call needs beyond its operands."""

    def _configure(self, problem, rtol, atol, max_steps):
        """Record the solve configuration. Must run before `HiPrim.__init__`, which turns
        `params` into attributes."""
        self.params = {
            "_problem": problem,
            "rtol": rtol,
            "atol": atol,
            "max_steps": max_steps,
        }

    def _kw(self):
        return {
            "problem": self._problem,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }

    def _call_ffi(self, target, kind, args, nbatch, ndirections=1):
        """Compile the callbacks this solve needs, then make the FFI call. Returns a tuple when
        the primitive has several outputs, and the one output otherwise."""
        self._problem._model.ensure_callbacks(kind, nbatch, ndirections)
        many = isinstance(self.out_aval, tuple)
        avals = self.out_aval if many else (self.out_aval,)
        out_types = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
        outs = jax.ffi.ffi_call(target, out_types)(
            *args,
            model_id=np.int64(self._problem._model.id),
            rtol=np.float64(self.rtol),
            atol=np.float64(self.atol),
            max_steps=np.int64(self.max_steps),
        )
        return tuple(outs) if many else outs[0]


class _LaneOp(_DiffsolPrim):
    """An operation on (params [*B, np], t0 f64[], t_eval [nt]) whose vmapped `params` axis
    becomes more lanes: the axis goes to the front, and the operation runs once for all the
    lanes. `t0` and `t_eval` are the same for every lane.

    Subclasses give the FFI target, the solve kind the callbacks are compiled for, and the
    shapes of the outputs."""

    target: str
    kind: str

    def __init__(
        self,
        params_aval,
        t0_aval,
        t_eval_aval,
        *,
        problem,
        rtol,
        atol,
        max_steps,
    ):
        self.in_avals = (params_aval, t0_aval, t_eval_aval)
        self.out_aval = self._out_aval(params_aval, t_eval_aval, problem)
        self._configure(problem, rtol, atol, max_steps)
        super().__init__()

    def _nbatch(self):
        return math.prod(self.in_avals[0].shape[:-1])

    def expand(self, params, t0, t_eval):
        return self._call_ffi(
            self.target, self.kind, (params, t0, t_eval), self._nbatch()
        )

    def batch(self, axis_data, args, dims):
        params, t0, t_eval = args
        pdim, t0dim, tdim = dims
        if t0dim is not None or tdim is not None:
            raise NotImplementedError(
                "diffsol_jax: vmap over t0, t1 or saveat is not supported"
            )
        if pdim is None:  # the rule is called even when no argument is batched
            return self(params, t0, t_eval), jax.tree.map(lambda _: None, self.out_aval)
        (params,) = bdims_at_front(axis_data, (params,), (pdim,))
        prim = type(self)(
            jax.typeof(params), self.in_avals[1], self.in_avals[2], **self._kw()
        )
        return prim(params, t0, t_eval), jax.tree.map(lambda _: 0, self.out_aval)


class _VjpFwdOp(_LaneOp):
    """The forward solve of the adjoint: -> (ys, handle s64[*B]). The checkpoint stays in C++;
    `handle` names it in the registry."""

    target, kind = "diffsol_solve_vjp_fwd", "vjp_fwd"

    def _out_aval(self, params_aval, t_eval_aval, problem):
        return (
            _ys_aval(params_aval, t_eval_aval, problem),
            ShapedArray(params_aval.shape[:-1], jnp.int64),
        )


class _FwdSensOp(_LaneOp):
    """The solve with forward sensitivities: -> (ys, sens [*B, np, nt, nout])."""

    target, kind = "diffsol_solve_fwd_sens", "fwd_sens"

    def _out_aval(self, params_aval, t_eval_aval, problem):
        ys = _ys_aval(params_aval, t_eval_aval, problem)
        return (
            ys,
            ShapedArray(
                params_aval.shape[:-1] + (params_aval.shape[-1],) + ys.shape[-2:],
                params_aval.dtype,
            ),
        )


class _Solve(_LaneOp):
    """(params [*B, np], t_eval [nt]) -> ys [*B, nt, nout]."""

    target, kind = "diffsol_solve_dense", "dense"

    def _out_aval(self, params_aval, t_eval_aval, problem):
        return _ys_aval(params_aval, t_eval_aval, problem)

    def jvp(self, primals, tangents):
        params, t0, t_eval = primals
        dparams, dt0, dt = tangents
        if not isinstance(dt0, Zero) or not isinstance(dt, Zero):
            raise NotImplementedError(
                "diffsol_jax: no derivative with respect to t0, t1 or saveat"
            )
        op = _FwdSensOp(
            jax.typeof(params), jax.typeof(t0), jax.typeof(t_eval), **self._kw()
        )
        ys, sens = op(params, t0, t_eval)
        return ys, jnp.einsum("...jtk,...j->...tk", sens, dparams)


class SolveAdjoint(_Solve):
    """Reverse mode with the diffsol adjoint. The backward solve also gets `params` and
    `t_eval`, so that it can run the forward solve again if the registry has freed the
    checkpoint of `handle`."""

    def vjp_fwd(self, nzs_in, params, t0, t_eval):
        # a cotangent of `None` for `t0` and `t_eval` would be a silent zero gradient
        if nzs_in[1] or nzs_in[2]:
            raise NotImplementedError(
                "diffsol_jax: no derivative with respect to t0, t1 or saveat"
            )
        op = _VjpFwdOp(
            jax.typeof(params), jax.typeof(t0), jax.typeof(t_eval), **self._kw()
        )
        ys, handle = op(params, t0, t_eval)
        return ys, (handle, params, t0, t_eval)

    def vjp_bwd_retval(self, res, g):
        if isinstance(g, Zero):
            return (None, None, None)
        handle, params, t0, t_eval = res
        prim = VjpBkwd(
            jax.typeof(handle),
            jax.typeof(params),
            jax.typeof(t0),
            jax.typeof(t_eval),
            jax.typeof(g),
            **self._kw(),
        )
        return prim(handle, params, t0, t_eval, g), None, None


class SolveForward(_Solve):
    """Reverse mode as the transpose of the forward sensitivities."""

    vjp_fwd, vjp_bwd_retval = vjp_from_jvp


class VjpBkwd(_DiffsolPrim):
    """The backward solve: (handle [*B], params [*B, np], t0 f64[], t_eval [nt],
    dgdu [*B, *D, nt, nout]) -> grad [*B, *D, np]. The axes of `dgdu` between the lanes and the
    last two are directions: the backward solve does all the directions together."""

    def __init__(
        self,
        handle_aval,
        params_aval,
        t0_aval,
        t_eval_aval,
        dgdu_aval,
        *,
        problem,
        rtol,
        atol,
        max_steps,
    ):
        self.in_avals = (handle_aval, params_aval, t0_aval, t_eval_aval, dgdu_aval)
        self.out_aval = ShapedArray(
            dgdu_aval.shape[:-2] + (params_aval.shape[-1],), params_aval.dtype
        )
        self._configure(problem, rtol, atol, max_steps)
        super().__init__()

    def expand(self, handle, params, t0, t_eval, dgdu):
        return self._call_ffi(
            "diffsol_solve_vjp_bkwd",
            "vjp_bkwd",
            (handle, params, t0, t_eval, dgdu),
            math.prod(handle.shape),
            math.prod(dgdu.shape[handle.ndim : -2]),
        )

    def _bind(self, handle, params, t0, t_eval, dgdu):
        prim = VjpBkwd(
            jax.typeof(handle),
            jax.typeof(params),
            jax.typeof(t0),
            jax.typeof(t_eval),
            jax.typeof(dgdu),
            **self._kw(),
        )
        return prim(handle, params, t0, t_eval, dgdu)

    def batch(self, axis_data, args, dims):
        """A vmapped axis of `handle` or `params` is a lane axis: it goes to the front, on all
        the arguments. A vmapped axis of only `dgdu` is a direction axis: it goes after the lane
        axes."""
        handle, params, t0, t_eval, dgdu = args
        hdim, pdim, t0dim, tdim, gdim = dims
        if t0dim is not None or tdim is not None:
            raise NotImplementedError(
                "diffsol_jax: vmap over t0, t1 or saveat is not supported"
            )
        if hdim is None and pdim is None and gdim is None:
            return self(handle, params, t0, t_eval, dgdu), None
        if hdim is not None or pdim is not None:
            handle, params, dgdu = bdims_at_front(
                axis_data, (handle, params, dgdu), (hdim, pdim, gdim)
            )
            return self._bind(handle, params, t0, t_eval, dgdu), 0
        pos = self.in_avals[0].ndim
        dgdu = jnp.moveaxis(dgdu, gdim, pos)
        return self._bind(handle, params, t0, t_eval, dgdu), pos
