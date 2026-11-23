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
    return ShapedArray(
        params_aval.shape[:-1] + (t_eval_aval.shape[0], problem.nsolution),
        params_aval.dtype,
    )


class _DiffsolPrim(HiPrim):
    def _configure(self, problem, rtol, atol, max_steps):
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
    becomes more lanes."""

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
    target, kind = "diffsol_solve_vjp_fwd", "vjp_fwd"

    def _out_aval(self, params_aval, t_eval_aval, problem):
        return (
            _ys_aval(params_aval, t_eval_aval, problem),
            ShapedArray(params_aval.shape[:-1], jnp.int64),
        )


class _FwdSensOp(_LaneOp):
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
    def vjp_fwd(self, nzs_in, params, t0, t_eval):
        # a cotangent of `None` for `t0` and `t_eval` would be a zero gradient
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
    vjp_fwd, vjp_bwd_retval = vjp_from_jvp


class VjpBkwd(_DiffsolPrim):
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
        """A vmapped axis of `handle` or `params` is a lane axis"""
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
