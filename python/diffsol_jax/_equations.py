import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._layout import Layout
from ._types import F32, F64, DiffsolJaxError

_LINEARITY_TOL = {F32: 1e-4, F64: 1e-9}


class CallbackId(enum.IntEnum):
    RHS = 0
    RHS_JAC = 1
    RHS_SENS = 2
    RHS_ADJOINT = 3
    RHS_SENS_ADJOINT = 4
    INIT = 5
    INIT_SENS = 6
    INIT_SENS_ADJOINT = 7
    MASS = 8
    MASS_ADJOINT = 9
    OUT = 10
    OUT_JAC = 11
    OUT_SENS = 12
    OUT_ADJOINT = 13
    OUT_SENS_ADJOINT = 14


LANE_ROLES = ("x", "p")
FULL_ROLES = ("v", "y")
SCALAR_ROLES = ("t", "beta")


@dataclass(frozen=True)
class Callback:
    id: CallbackId
    fn: Callable
    args: tuple[str, ...]
    nv: int
    ny: int


@dataclass(frozen=True)
class Equations:
    rhs: Callable
    init: Callable
    nstates: int
    nparams: int
    dtype: Any
    out: Callable | None = None
    nout: int = 0
    mass: Callable | None = None

    def callbacks(self) -> dict[CallbackId, Callback]:
        return _callbacks(self)

    def available(self) -> list[CallbackId]:
        return list(_callbacks(self))


def _at(f, i, args):
    return lambda a: f(*args[:i], a, *args[i + 1 :])


def _jvp(f, i):
    def fn(*args):
        *primals, v = args
        return jax.jvp(_at(f, i, primals), (primals[i],), (v,))[1]

    return fn


def _vjp(f, i):
    def fn(*args):
        *primals, v = args
        return -jax.vjp(_at(f, i, primals), primals[i])[1](v)[0]

    return fn


_DERIVATIONS = (
    ("", None, None),
    ("_JAC", "x", _jvp),
    ("_SENS", "p", _jvp),
    ("_ADJOINT", "x", _vjp),
    ("_SENS_ADJOINT", "p", _vjp),
)


def _derived(eq, name, f, args, nresult):
    width = {"x": eq.nstates, "p": eq.nparams}
    for suffix, wrt, mode in _DERIVATIONS:
        if wrt is None:
            yield Callback(CallbackId[name], f, args, 0, nresult)
        elif wrt in args:
            nv, ny = (width[wrt], nresult) if mode is _jvp else (nresult, width[wrt])
            yield Callback(
                CallbackId[name + suffix],
                mode(f, args.index(wrt)),
                args + ("v",),
                nv,
                ny,
            )


def _mass(eq):
    """`mass` and `mass_adjoint`, which diffsol calls as `y = M(p, t) v + beta y`."""
    m, n = eq.mass, eq.nstates
    args = ("v", "p", "t", "beta", "y")
    yield Callback(
        CallbackId.MASS, lambda v, p, t, beta, y: m(v, p, t) + beta * y, args, n, n
    )
    # `M^T v`, from the VJP of the linear map `u -> M u`
    yield Callback(
        CallbackId.MASS_ADJOINT,
        lambda v, p, t, beta, y: jax.vjp(lambda u: m(u, p, t), v)[1](v)[0] + beta * y,
        args,
        n,
        n,
    )


def _callbacks(eq) -> dict[CallbackId, Callback]:
    sources = (
        ("RHS", eq.rhs, ("x", "p", "t"), eq.nstates),
        ("INIT", eq.init, ("p", "t"), eq.nstates),
        ("OUT", eq.out, ("x", "p", "t"), eq.nout),
    )
    table = {}
    for name, f, args, nresult in sources:
        if f is not None:
            table.update({cb.id: cb for cb in _derived(eq, name, f, args, nresult)})
    if eq.mass is not None:
        table.update({cb.id: cb for cb in _mass(eq)})
    return table


def over_flat(f, name, state: Layout, params: Layout, result: Layout):
    if f is None:
        return None

    def wrapped(a, p, t):
        y = f(state.unravel(a), params.unravel(p), t)
        return result.ravel(y, f"the result of {name}")

    return wrapped


def diagonal_mass(mask, dtype):
    if all(mask) or not any(mask):
        raise DiffsolJaxError("algebraic must mark some, not all, of the state")
    diff = jnp.array([0.0 if a else 1.0 for a in mask], dtype)
    return lambda v, p, t: diff * v


def check_mass_is_linear(mass, nstates, nparams, dtype):
    """Check at a random point that `mass(v, p, t)` is linear in `v` and independent of `p`."""
    rng = np.random.default_rng(0)
    v1, v2 = (
        jnp.asarray(rng.standard_normal(nstates), dtype),
        jnp.asarray(rng.standard_normal(nstates), dtype),
    )
    p1, p2 = (
        jnp.asarray(rng.standard_normal(nparams), dtype),
        jnp.asarray(rng.standard_normal(nparams), dtype),
    )
    t = 0.37
    a, b = 2.3, -1.7
    tol = _LINEARITY_TOL[dtype]
    lhs = mass(a * v1 + b * v2, p1, t)
    rhs = a * mass(v1, p1, t) + b * mass(v2, p1, t)
    if not np.allclose(np.asarray(lhs), np.asarray(rhs), rtol=tol, atol=tol):
        raise DiffsolJaxError(
            "mass(v, p, t) must be linear in v (the action of a mass matrix)"
        )
    if not np.allclose(
        np.asarray(mass(v1, p1, t)), np.asarray(mass(v1, p2, t)), rtol=tol
    ):
        raise DiffsolJaxError(
            "mass(v, p, t) must not depend on p: the diffsol sensitivities have no dM/dp term, "
            "so the gradient would be wrong. Fold the parameter into rhs instead (for example "
            "divide the differential equations by it, or scale the algebraic equations)"
        )


def solution_layout(out, state: Layout, params: Layout, dtype) -> Layout:
    if out is None:
        return state

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, dtype)

    traced = jax.eval_shape(
        lambda a, p, t: out(state.unravel(a), params.unravel(p), t),
        spec(state.n),
        spec(params.n),
        spec(),
    )
    return Layout(traced, "out", dtype, constants=False)


def build(rhs, out, flat_mass, state: Layout, params: Layout, solution: Layout, dtype):
    nparams = params.n

    def over_params(f):
        return None if f is None else lambda a, p, t: f(a, p[:nparams], t)

    return Equations(
        rhs=over_params(over_flat(rhs, "rhs", state, params, state)),
        init=lambda p, t: p[nparams:],
        nstates=state.n,
        nparams=nparams + state.n,
        dtype=dtype,
        out=over_params(over_flat(out, "out", state, params, solution)),
        nout=0 if out is None else solution.n,
        mass=over_params(flat_mass),
    )
