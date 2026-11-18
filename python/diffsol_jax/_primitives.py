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


def _ffi(target, out_avals, *args, problem, rtol, atol):
    out_types = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in out_avals]
    return jax.ffi.ffi_call(target, out_types)(
        *args,
        model_id=np.int64(problem._id),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
    )


def _f64(shape):
    return ShapedArray(shape, jnp.float64)


def _ys_aval(params_aval, t_eval_aval, problem):
    return _f64(params_aval.shape[:-1] + (t_eval_aval.shape[0], problem.nsolution))


class _LaneOp(HiPrim):
    """An operation on (params [*B, np], t_eval [nt]) whose vmapped `params` axis becomes more
    lanes: the axis goes to the front, and the operation runs once for all the lanes."""

    def __init__(self, params_aval, t_eval_aval, *, problem, rtol, atol):
        self.in_avals = (params_aval, t_eval_aval)
        self.out_aval = self._out_aval(params_aval, t_eval_aval, problem)
        self.params = {"_problem": problem, "rtol": rtol, "atol": atol}
        super().__init__()

    def _kw(self):
        return {"problem": self._problem, "rtol": self.rtol, "atol": self.atol}

    def _nbatch(self):
        return math.prod(self.in_avals[0].shape[:-1])

    def batch(self, axis_data, args, dims):
        params, t_eval = args
        pdim, tdim = dims
        if tdim is not None:
            raise NotImplementedError("diffsol_jax: vmap over t_eval is not supported")
        if pdim is None:  # the rule is called even when no argument is batched
            return self(params, t_eval), jax.tree.map(lambda _: None, self.out_aval)
        (params,) = bdims_at_front(axis_data, (params,), (pdim,))
        prim = type(self)(jax.typeof(params), self.in_avals[1], **self._kw())
        return prim(params, t_eval), jax.tree.map(lambda _: 0, self.out_aval)


class _VjpFwdOp(_LaneOp):
    """The forward solve of the adjoint: -> (ys, handle s64[*B]). The checkpoint stays in C++;
    `handle` names it in the registry."""

    def _out_aval(self, params_aval, t_eval_aval, problem):
        return (
            _ys_aval(params_aval, t_eval_aval, problem),
            ShapedArray(params_aval.shape[:-1], jnp.int64),
        )

    def expand(self, params, t_eval):
        self._problem._ensure_callbacks("vjp_fwd", self._nbatch())
        ys, handle = _ffi(
            "diffsol_solve_vjp_fwd", self.out_aval, params, t_eval, **self._kw()
        )
        return ys, handle


class _FwdSensOp(_LaneOp):
    """The solve with forward sensitivities: -> (ys, sens [*B, np, nt, nout])."""

    def _out_aval(self, params_aval, t_eval_aval, problem):
        ys = _ys_aval(params_aval, t_eval_aval, problem)
        return (
            ys,
            _f64(params_aval.shape[:-1] + (params_aval.shape[-1],) + ys.shape[-2:]),
        )

    def expand(self, params, t_eval):
        self._problem._ensure_callbacks("fwd_sens", self._nbatch())
        ys, sens = _ffi(
            "diffsol_solve_fwd_sens", self.out_aval, params, t_eval, **self._kw()
        )
        return ys, sens


class _Solve(_LaneOp):
    """(params [*B, np], t_eval [nt]) -> ys [*B, nt, nout]."""

    def _out_aval(self, params_aval, t_eval_aval, problem):
        return _ys_aval(params_aval, t_eval_aval, problem)

    def expand(self, params, t_eval):
        self._problem._ensure_callbacks("dense", self._nbatch())
        (ys,) = _ffi(
            "diffsol_solve_dense", [self.out_aval], params, t_eval, **self._kw()
        )
        return ys

    def jvp(self, primals, tangents):
        params, t_eval = primals
        dparams, dt = tangents
        if not isinstance(dt, Zero):
            raise NotImplementedError(
                "diffsol_jax: no derivative with respect to t_eval"
            )
        op = _FwdSensOp(jax.typeof(params), jax.typeof(t_eval), **self._kw())
        ys, sens = op(params, t_eval)
        return ys, jnp.einsum("...jtk,...j->...tk", sens, dparams)


class SolveAdjoint(_Solve):
    """Reverse mode with the diffsol adjoint. The backward solve also gets `params` and
    `t_eval`, so that it can run the forward solve again if the registry has freed the
    checkpoint of `handle`."""

    def vjp_fwd(self, nzs_in, params, t_eval):
        del nzs_in
        op = _VjpFwdOp(jax.typeof(params), jax.typeof(t_eval), **self._kw())
        ys, handle = op(params, t_eval)
        return ys, (handle, params, t_eval)

    def vjp_bwd_retval(self, res, g):
        if isinstance(g, Zero):
            return (None, None)
        handle, params, t_eval = res
        prim = VjpBkwd(
            jax.typeof(handle),
            jax.typeof(params),
            jax.typeof(t_eval),
            jax.typeof(g),
            **self._kw(),
        )
        return prim(handle, params, t_eval, g), None


class SolveForward(_Solve):
    """Reverse mode as the transpose of the forward sensitivities."""

    vjp_fwd, vjp_bwd_retval = vjp_from_jvp


class VjpBkwd(HiPrim):
    """The backward solve: (handle [*B], params [*B, np], t_eval [nt], dgdu [*B, *D, nt, nout])
    -> grad [*B, *D, np]. The axes of `dgdu` between the lanes and the last two are directions:
    the backward solve does all the directions together."""

    def __init__(
        self, handle_aval, params_aval, t_eval_aval, dgdu_aval, *, problem, rtol, atol
    ):
        self.in_avals = (handle_aval, params_aval, t_eval_aval, dgdu_aval)
        self.out_aval = _f64(dgdu_aval.shape[:-2] + (params_aval.shape[-1],))
        self.params = {"_problem": problem, "rtol": rtol, "atol": atol}
        super().__init__()

    def _kw(self):
        return {"problem": self._problem, "rtol": self.rtol, "atol": self.atol}

    def expand(self, handle, params, t_eval, dgdu):
        nbatch = math.prod(handle.shape)
        ndirections = math.prod(dgdu.shape[handle.ndim : -2])
        self._problem._ensure_callbacks("vjp_bkwd", nbatch, ndirections)
        (grad,) = _ffi(
            "diffsol_solve_vjp_bkwd",
            [self.out_aval],
            handle,
            params,
            t_eval,
            dgdu,
            **self._kw(),
        )
        return grad

    def _bind(self, handle, params, t_eval, dgdu):
        prim = VjpBkwd(
            jax.typeof(handle),
            jax.typeof(params),
            jax.typeof(t_eval),
            jax.typeof(dgdu),
            **self._kw(),
        )
        return prim(handle, params, t_eval, dgdu)

    def batch(self, axis_data, args, dims):
        """A vmapped axis of `handle` or `params` is a lane axis: it goes to the front, on all
        the arguments. A vmapped axis of only `dgdu` is a direction axis: it goes after the lane
        axes."""
        handle, params, t_eval, dgdu = args
        hdim, pdim, tdim, gdim = dims
        if tdim is not None:
            raise NotImplementedError("diffsol_jax: vmap over t_eval is not supported")
        if hdim is None and pdim is None and gdim is None:
            return self(handle, params, t_eval, dgdu), None
        if hdim is not None or pdim is not None:
            handle, params, dgdu = bdims_at_front(
                axis_data, (handle, params, dgdu), (hdim, pdim, gdim)
            )
            return self._bind(handle, params, t_eval, dgdu), 0
        pos = self.in_avals[0].ndim
        dgdu = jnp.moveaxis(dgdu, gdim, pos)
        return self._bind(handle, params, t_eval, dgdu), pos
