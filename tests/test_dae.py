"""Tests of DAE support: a singular mass matrix gives a semi-explicit index-1 DAE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
TOL = {"rtol": 1e-10, "atol": 1e-10}
P = jnp.array([0.7, 2.0])


def y0_of(p):
    """The consistent initial state: y(0) = 1 and z(0) = p1 y(0)."""
    return jnp.array([1.0, p[1]])


def linear_dae_rhs(x, p, t):
    """y' = -p0 y and 0 = z - p1 y, so y = exp(-p0 t) and z = p1 y."""
    return jnp.array([-p[0] * x[0], x[1] - p[1] * x[0]])


def dae_problem(**kw):
    kw.setdefault("mass", lambda v, p, t: jnp.array([v[0], 0.0]))
    return ODEProblem(linear_dae_rhs, state=2, params=2, **kw)


DIAGONAL_MASS = dae_problem()


def linear_dae(y0, p, problem=DIAGONAL_MASS, **kw):
    return solve(problem, y0, T_EVAL, p, **{**TOL, **kw})


def exact(p, t):
    y = jnp.exp(-p[0] * t)
    return jnp.stack([y, p[1] * y], axis=-1)


def finite_difference(f, p, eps=1e-6):
    basis = jnp.eye(p.shape[0])
    return jnp.array([(f(p + eps * e) - f(p - eps * e)) / (2 * eps) for e in basis])


@pytest.mark.parametrize("solver", ["bdf", "esdirk34", "tr_bdf2"])
def test_dae_matches_analytic(solver):
    ys = linear_dae(y0_of(P), P, dae_problem(solver=solver))
    np.testing.assert_allclose(ys, exact(P, T_EVAL), rtol=1e-6)


def test_algebraic_shortcut_matches_mass():
    ys = linear_dae(y0_of(P), P)
    for algebraic in (1, [False, True], np.array([False, True])):
        problem = dae_problem(mass=None, algebraic=algebraic)
        np.testing.assert_allclose(linear_dae(y0_of(P), P, problem), ys, rtol=1e-9)


def test_inconsistent_y0_is_projected():
    # y0 gives z(0) = 99, but z(0) = p1 y(0) = 2: diffsol solves the algebraic variables at t0
    y0 = jnp.array([1.0, 99.0])
    np.testing.assert_allclose(linear_dae(y0, P), exact(P, T_EVAL), rtol=1e-6)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_dae_grad_matches_finite_difference(ad):
    def loss(p):
        return linear_dae(y0_of(p), p, ad=ad).sum()

    grad = jax.jit(jax.grad(loss))(P)
    np.testing.assert_allclose(grad, finite_difference(loss, P), rtol=1e-5)


def test_dae_jacfwd_matches_jacrev():
    def f(p):
        return linear_dae(y0_of(p), p)

    np.testing.assert_allclose(
        jax.jacfwd(f)(P), jax.jacrev(f)(P), rtol=1e-6, atol=1e-10
    )


def test_dae_vmap_of_grad_matches_loop():
    params = jnp.array([[0.7, 2.0], [1.1, 0.5], [0.3, 3.0]])

    def loss(p):
        return linear_dae(y0_of(p), p).sum()

    grads = jax.jit(jax.vmap(jax.grad(loss)))(params)
    loop = jnp.stack([jax.grad(loss)(p) for p in params])
    np.testing.assert_allclose(grads, loop, rtol=1e-6)


def robertson(dae):
    """The Robertson problem, as the ODE, or as the DAE with the conservation law
    0 = x + y + z - 1 in place of the third rate equation."""

    def rhs(x, p, t):
        k1, k2, k3 = p
        r1, r2 = k1 * x[0] - k3 * x[1] * x[2], k2 * x[1] ** 2
        last = x[0] + x[1] + x[2] - 1.0 if dae else r2
        return jnp.array([-r1, r1 - r2, last])

    return ODEProblem(rhs, state=3, params=3, algebraic=1 if dae else None)


ROBERTSON_TOL = {"rtol": 1e-10, "atol": 1e-12}


def test_robertson_dae_matches_ode_form():
    p = jnp.array([0.04, 3e7, 1e4])
    t = jnp.array([0.0, 0.4, 4.0, 40.0, 400.0])
    ode, dae = robertson(dae=False), robertson(dae=True)
    y0 = jnp.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        solve(dae, y0, t, p, **ROBERTSON_TOL),
        solve(ode, y0, t, p, **ROBERTSON_TOL),
        rtol=1e-6,
        atol=1e-10,
    )

    # the adjoint of the DAE form against the adjoint of the ODE form; y2 is of order 1e-5,
    # so the gradient entries span many orders of magnitude
    def loss(problem):
        return lambda q: (
            solve(problem, y0, t, q * p, **ROBERTSON_TOL)[:, 1] * 1e4
        ).sum()

    one = jnp.ones(3)
    np.testing.assert_allclose(
        jax.grad(loss(dae))(one), jax.grad(loss(ode))(one), rtol=1e-4
    )


def test_time_dependent_mass():
    # (1 + t) y' = -p0 y: y = (1 + t) ** -p0
    problem = ODEProblem(
        rhs=lambda x, p, t: -p[0] * x,
        state=1,
        params=2,
        mass=lambda v, p, t: (1.0 + t) * v,
    )
    y0 = jnp.array([1.0])
    ys = solve(problem, y0, T_EVAL, P, **TOL)
    np.testing.assert_allclose(ys[:, 0], (1.0 + T_EVAL) ** -P[0], rtol=1e-6)

    def loss(p):
        return solve(problem, y0, T_EVAL, p, **TOL).sum()

    np.testing.assert_allclose(jax.grad(loss)(P), finite_difference(loss, P), rtol=1e-5)


def test_parameter_dependent_mass_raises():
    # the diffsol sensitivity and adjoint equations have no dM/dp term, so the gradient of a
    # parameter-dependent mass would silently miss it (checked against finite differences)
    with pytest.raises(DiffsolJaxError, match="dM/dp"):
        ODEProblem(
            rhs=lambda x, p, t: -p[0] * x,
            state=1,
            params=2,
            mass=lambda v, p, t: p[1] * v,
        )


def test_explicit_solver_raises():
    with pytest.raises(DiffsolJaxError, match="implicit"):
        dae_problem(solver="tsit45")


def test_nonlinear_mass_raises():
    with pytest.raises(DiffsolJaxError, match="linear"):
        dae_problem(mass=lambda v, p, t: v**2)


def test_bad_algebraic_raises():
    with pytest.raises(DiffsolJaxError, match="not both"):
        dae_problem(algebraic=1)  # dae_problem also sets mass
    with pytest.raises(DiffsolJaxError, match="not a bool"):
        dae_problem(mass=None, algebraic=True)
    with pytest.raises(DiffsolJaxError, match="algebraic must be in"):
        dae_problem(mass=None, algebraic=2)  # all the states algebraic
    with pytest.raises(DiffsolJaxError, match="must have length 2"):
        dae_problem(mass=None, algebraic=[True])
