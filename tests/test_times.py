"""Tests of the times of a solve: `y0` is the state at `t0`, the solve ends at `t1`, and
`saveat` chooses the times it returns."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

TOL = {"rtol": 1e-11, "atol": 1e-11}
Y0 = jnp.array([1.0])
P = jnp.array([1.0])
# y' = p y, so y(t) = y0 exp(p (t - t0))
EXPONENTIAL = ODEProblem(lambda y, p, t: p[0] * y, state=1, params=1)
# y' = t, so y(t) = y0 + (t^2 - t0^2) / 2: an rhs that reads `t` itself
RAMP = ODEProblem(lambda y, p, t: 0.0 * y + t, state=1, params=None)


def exponential(t0, t1, **kw):
    return solve(EXPONENTIAL, Y0, P, t0, t1, **{**TOL, **kw})


def test_the_default_returns_t1_alone():
    ys = exponential(0.0, 2.0)
    assert ys.shape == (1, 1)
    np.testing.assert_allclose(ys.ravel(), [np.exp(2.0)], rtol=1e-9)


def test_y0_is_the_state_at_t0():
    """The solution is `exp(t - t0)`, not `exp(t)`: `y0` belongs to `t0`."""
    saveat = jnp.array([1.0, 2.0, 3.0])
    ys = exponential(1.0, 3.0, saveat=saveat)
    np.testing.assert_allclose(ys.ravel(), np.exp([0.0, 1.0, 2.0]), rtol=1e-9)


def test_saveat_returns_every_time_it_asks_for():
    saveat = jnp.linspace(0.0, 2.0, 5)
    ys = exponential(0.0, 2.0, saveat=saveat)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys.ravel(), np.exp(np.asarray(saveat)), rtol=1e-9)


def test_saveat_can_start_after_t0():
    """The solve runs from `t0`; the window it returns can be anywhere in `[t0, t1]`."""
    ys = exponential(0.0, 4.0, saveat=jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(ys.ravel(), np.exp([2.0, 3.0]), rtol=1e-9)


def test_the_equations_see_the_true_time():
    """`t0` moves the start of the solve, not the clock the equations read."""
    ys = solve(RAMP, jnp.array([0.0]), None, 2.0, 3.0, **TOL)
    np.testing.assert_allclose(ys.ravel(), [2.5], atol=1e-9)


def test_t0_and_t1_can_be_traced():
    f = jax.jit(lambda t0, t1: exponential(t0, t1))
    np.testing.assert_allclose(f(1.0, 3.0).ravel(), [np.exp(2.0)], rtol=1e-9)


def test_lanes_share_the_times():
    out = jax.vmap(lambda p: solve(EXPONENTIAL, Y0, p, 0.0, 1.0, **TOL))(
        jnp.array([[1.0], [2.0]])
    )
    assert out.shape == (2, 1, 1)
    np.testing.assert_allclose(out.ravel(), np.exp([1.0, 2.0]), rtol=1e-9)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_gradients_are_taken_from_t0(ad):
    saveat = jnp.array([1.0, 2.0, 3.0])

    def f(y0, p):
        return solve(EXPONENTIAL, y0, p, 1.0, 3.0, saveat=saveat, ad=ad, **TOL).sum()

    g_y0, g_p = jax.grad(f, (0, 1))(Y0, P)
    # sum over t = 1, 2, 3 of y0 exp(p (t - 1))
    np.testing.assert_allclose(g_y0.ravel(), [1.0 + np.e + np.e**2], rtol=1e-7)
    np.testing.assert_allclose(g_p.ravel(), [np.e + 2 * np.e**2], rtol=1e-7)


def test_the_checkpoint_of_one_t0_is_not_used_for_another():
    """`t0` is part of the key of an adjoint checkpoint, so two solves that differ only in it
    do not share one."""

    def grad_at(t0):
        saveat = jnp.array([1.0, 2.0])
        return jax.grad(
            lambda p: solve(EXPONENTIAL, Y0, p, t0, 2.0, saveat=saveat, **TOL).sum()
        )(P)

    # from t0 = 0: d/dp [e^p + e^2p] = e + 2 e^2; from t0 = 1: d/dp [1 + e^p] = e
    np.testing.assert_allclose(grad_at(0.0).ravel(), [np.e + 2 * np.e**2], rtol=1e-7)
    np.testing.assert_allclose(grad_at(1.0).ravel(), [np.e], rtol=1e-7)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"t0": 1.0, "t1": 0.0}, "t1 must be after t0"),
        ({"t0": 2.0, "t1": 2.0}, "t1 must be after t0"),
        ({"t0": 1.0, "t1": 3.0, "saveat": jnp.array([0.0, 2.0])}, "at or after t0"),
        ({"t0": 0.0, "t1": 1.0, "saveat": jnp.array([0.5, 2.0])}, "at or before t1"),
        ({"t0": 0.0, "t1": 2.0, "saveat": jnp.array([1.0, 1.0])}, "must be increasing"),
        ({"t0": 0.0, "t1": 2.0, "saveat": jnp.array([0.0])}, "must reach past t0"),
        ({"t0": 0.0, "t1": 2.0, "saveat": jnp.array([])}, "at least one time"),
        ({"t0": jnp.zeros(2), "t1": 1.0}, "must be scalars"),
    ],
)
def test_time_argument_errors(kwargs, match):
    with pytest.raises(DiffsolJaxError, match=match):
        solve(EXPONENTIAL, Y0, P, **kwargs)


def test_no_vmap_over_the_times():
    with pytest.raises(NotImplementedError, match="vmap over t0, t1 or saveat"):
        jax.vmap(lambda t0: exponential(t0, 3.0))(jnp.array([0.0, 1.0]))


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_no_gradient_through_the_times(ad):
    """A zero cotangent for `t0` would be a silent wrong answer, so it is an error."""
    with pytest.raises(NotImplementedError, match="with respect to t0, t1 or saveat"):
        jax.grad(lambda t0: exponential(t0, 3.0, ad=ad).sum())(0.0)
