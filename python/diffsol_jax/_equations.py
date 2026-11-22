"""The equation set the solver is built from: the user functions over flat vectors, and the
diffsol callbacks derived from them by AD."""

import enum
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._layout import Layout
from ._types import F64, DiffsolJaxError


class CallbackId(enum.IntEnum):
    """The callbacks the prototype supports (no `root` and `reset`: diffsol-c does not allow
    them with more than one lane). The values are the same as `CallbackId` in `callbacks.h`."""

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


# The roles of the callback arguments. A callback's argument list is a tuple of these, in the
# order the executable takes them:
#   x    [B, nstates]  the state          p     [B, nparams]  the parameters
#   t    []            the time           beta  []            the mass shift
#   v    [B, P, nv]    the vector a derived callback multiplies
#   y    [B, P, ny]    `mass`'s output, which is also an input
# `x` and `p` are the same for every lane on the `P` axis; `v` and `y` are not, so they alone
# carry it.
LANE_ROLES = ("x", "p")
FULL_ROLES = ("v", "y")
SCALAR_ROLES = ("t", "beta")


@dataclass(frozen=True)
class Callback:
    """One diffsol callback: the function of one lane, the roles of its arguments, and the
    widths of the `v` and `y` axes (`nv = 0` for a callback with no `v`)."""

    id: CallbackId
    fn: Callable
    args: tuple[str, ...]
    nv: int
    ny: int


@dataclass(frozen=True)
class Equations:
    """The user equations as functions of the flat state and parameter vectors."""

    rhs: Callable
    init: Callable
    nstates: int
    nparams: int
    out: Callable | None = None
    nout: int = 0
    mass: Callable | None = None

    def callbacks(self) -> dict[CallbackId, Callback]:
        return _callbacks(self)

    def available(self) -> list[CallbackId]:
        return list(_callbacks(self))


# ---- deriving the callbacks ------------------------------------------------------------------


def _at(f, i, args):
    """`f` as a function of argument `i` alone, with the others fixed at `args`."""
    return lambda a: f(*args[:i], a, *args[i + 1 :])


def _jvp(f, i):
    """`(df/d args[i]) v`, with `v` the last argument."""

    def fn(*args):
        *primals, v = args
        return jax.jvp(_at(f, i, primals), (primals[i],), (v,))[1]

    return fn


def _vjp(f, i):
    """`-(df/d args[i])^T v`, with `v` the last argument. diffsol wants the negative for every
    adjoint callback."""

    def fn(*args):
        *primals, v = args
        return -jax.vjp(_at(f, i, primals), primals[i])[1](v)[0]

    return fn


# Each of the three source functions gets the plain callback and one callback for each way of
# differentiating it: the name suffix, the role differentiated with respect to, and the mode.
# `init` has no `x` argument, so it gets neither `_JAC` nor `_ADJOINT`.
_DERIVATIONS = (
    ("", None, None),
    ("_JAC", "x", _jvp),
    ("_SENS", "p", _jvp),
    ("_ADJOINT", "x", _vjp),
    ("_SENS_ADJOINT", "p", _vjp),
)


def _derived(eq, name, f, args, nresult):
    """The callbacks of one source function. A forward derivative multiplies something the size
    of the argument it differentiates and returns something the size of the result; a reverse
    one does the opposite."""
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
    # `M^T v` (not negated), from the VJP of the linear map `u -> M u`
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


# ---- building them from the user functions ---------------------------------------------------


def over_flat(f, name, state: Layout, params: Layout, result: Layout):
    """`f` over the pytrees, as a function of the flat vectors."""
    if f is None:
        return None

    def wrapped(a, p, t):
        y = f(state.unravel(a), params.unravel(p), t)
        return result.ravel(y, f"the result of {name}")

    return wrapped


def diagonal_mass(mask):
    """The diagonal mass of a boolean mask over the flat state, True denoting an algebraic
    entry: the rows of `rhs` it selects become algebraic equations."""
    if all(mask) or not any(mask):
        raise DiffsolJaxError("algebraic must mark some, not all, of the state")
    diff = jnp.array([0.0 if a else 1.0 for a in mask], F64)
    return lambda v, p, t: diff * v


def check_mass_is_linear(mass, nstates, nparams):
    """Check at a random point that `mass(v, p, t)` is linear in `v` and independent of `p`."""
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


def solution_layout(out, state: Layout, params: Layout) -> Layout:
    """What `ys` comes back in: the state, or the structure `out` returns, found by shape
    tracing."""
    if out is None:
        return state

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, F64)

    traced = jax.eval_shape(
        lambda a, p, t: out(state.unravel(a), params.unravel(p), t),
        spec(state.n),
        spec(params.n),
        spec(),
    )
    return Layout(traced, "out", constants=False)


def build(rhs, out, flat_mass, state: Layout, params: Layout, solution: Layout):
    """The equation set of a problem.

    diffsol's initial state is `init(p, t)`, a function of the parameters, so `y0` rides in the
    tail of the internal parameter vector `[params, y0]`. Then `init` is a slice, `init_sens` is
    the constant selection matrix `[0 | I]`, and the gradient with respect to `y0` comes out of
    the existing sensitivity and adjoint callbacks with no special case. The user equations only
    see the `params` head.

    The returned closures hold the layouts, never the problem: a reference from the equations
    back to the problem would be a cycle, and the native solver would then outlive the problem
    until the next gc pass.
    """
    nparams = params.n

    def over_params(f):
        return None if f is None else lambda a, p, t: f(a, p[:nparams], t)

    return Equations(
        rhs=over_params(over_flat(rhs, "rhs", state, params, state)),
        init=lambda p, t: p[nparams:],
        nstates=state.n,
        nparams=nparams + state.n,
        out=over_params(over_flat(out, "out", state, params, solution)),
        nout=0 if out is None else solution.n,
        mass=over_params(flat_mass),
    )
