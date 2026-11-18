"""Tests of the `state` layout of a problem: a flat `state=<int>`, or a pytree that `y0`, the
equations and `ys` all share."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.linspace(0.0, 2.0, 5)
TOL = {"rtol": 1e-10, "atol": 1e-10}
N = 3
# y'' = -k y as {"pos", "vel"}: y = pos0 cos(sqrt(k) t)
STATE = {"pos": jnp.zeros(N), "vel": jnp.zeros(N)}
SPRING = ODEProblem(
    lambda y, p, t: {"pos": y["vel"], "vel": -p[0] * y["pos"]},
    state=STATE,
    params=1,
)
POS0 = jnp.array([1.0, 0.0, 0.5])
Y0 = {"pos": POS0, "vel": jnp.zeros(N)}
K = jnp.array([4.0])


def exact(k, t):
    w = jnp.sqrt(k[0])
    return {
        "pos": jnp.outer(jnp.cos(w * t), POS0),
        "vel": -w * jnp.outer(jnp.sin(w * t), POS0),
    }


def test_flat_state_stays_one_array():
    """`state=<int>` is one `f64[n]` leaf, so `y0` and `ys` are plain arrays."""
    problem = ODEProblem(lambda y, p, t: -p[0] * y, state=2, params=1)
    ys = solve(problem, jnp.array([1.0, 2.0]), jnp.array([0.5]), T_EVAL, **TOL)
    assert isinstance(ys, jax.Array)
    assert ys.shape == (T_EVAL.shape[0], 2)
    np.testing.assert_allclose(
        ys, jnp.outer(jnp.exp(-0.5 * T_EVAL), jnp.array([1.0, 2.0])), rtol=1e-7
    )


def test_pytree_state_solve():
    assert (SPRING.nstates, SPRING.nparams) == (2 * N, 1)
    ys = solve(SPRING, Y0, K, T_EVAL, **TOL)
    assert set(ys) == {"pos", "vel"}
    assert ys["pos"].shape == (T_EVAL.shape[0], N)
    want = exact(K, T_EVAL)
    np.testing.assert_allclose(ys["pos"], want["pos"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(ys["vel"], want["vel"], rtol=1e-6, atol=1e-8)


def test_pytree_state_matches_the_flat_form():
    """The same system with the state raveled by hand gives the same solution."""
    flat = ODEProblem(
        lambda y, p, t: jnp.concatenate([y[N:], -p[0] * y[:N]]), state=2 * N, params=1
    )
    ys_flat = solve(flat, jnp.concatenate([POS0, jnp.zeros(N)]), K, T_EVAL, **TOL)
    ys = solve(SPRING, Y0, K, T_EVAL, **TOL)
    np.testing.assert_allclose(ys["pos"], ys_flat[:, :N], rtol=1e-12)
    np.testing.assert_allclose(ys["vel"], ys_flat[:, N:], rtol=1e-12)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_of_a_pytree_state_and_params(ad):
    def loss(y0, p):
        return solve(SPRING, y0, p, T_EVAL, ad=ad, **TOL)["pos"].sum()

    g_y0, g_k = jax.jit(jax.grad(loss, (0, 1)))(Y0, K)
    assert set(g_y0) == {"pos", "vel"}
    assert g_y0["pos"].shape == (N,)
    # d/dpos0 of sum(cos(w t) pos0) is sum(cos(w t)), the same for every entry
    w = jnp.sqrt(K[0])
    np.testing.assert_allclose(
        g_y0["pos"], jnp.full((N,), jnp.cos(w * T_EVAL).sum()), rtol=1e-6
    )
    h = 1e-6
    fd = (loss(Y0, K + h) - loss(Y0, K - h)) / (2 * h)
    np.testing.assert_allclose(g_k, fd, rtol=1e-5)


def test_vmap_over_a_state_leaf():
    """A vmapped leaf of `y0` is lanes of one batched solve, like any other vmapped input."""
    vels = jnp.stack([jnp.zeros(N), jnp.ones(N), -jnp.ones(N)])
    batched = jax.jit(
        jax.vmap(lambda v: solve(SPRING, {**Y0, "vel": v}, K, T_EVAL, **TOL)["pos"])
    )
    ys = batched(vels)
    loop = jnp.stack(
        [solve(SPRING, {**Y0, "vel": v}, K, T_EVAL, **TOL)["pos"] for v in vels]
    )
    np.testing.assert_allclose(ys, loop, rtol=1e-7)
    assert batched.lower(vels).as_text().count("diffsol_solve_dense") == 1


def test_out_returns_its_own_pytree():
    problem = ODEProblem(
        lambda y, p, t: {"pos": y["vel"], "vel": -p[0] * y["pos"]},
        state=STATE,
        params=1,
        out=lambda y, p, t: {
            "energy": 0.5 * (y["vel"] @ y["vel"] + p[0] * (y["pos"] @ y["pos"])),
            "first": y["pos"][:1],
        },
    )
    assert problem.nout == 2
    ys = solve(problem, Y0, K, T_EVAL, **TOL)
    assert set(ys) == {"energy", "first"}
    assert ys["energy"].shape == (T_EVAL.shape[0],)
    assert ys["first"].shape == (T_EVAL.shape[0], 1)
    # the energy of y'' = -k y is conserved
    e0 = 0.5 * K[0] * (POS0 @ POS0)
    np.testing.assert_allclose(ys["energy"], jnp.full(T_EVAL.shape, e0), rtol=1e-6)
    np.testing.assert_allclose(
        ys["first"][:, 0], exact(K, T_EVAL)["pos"][:, 0], rtol=1e-6, atol=1e-8
    )


def test_algebraic_as_a_mask_over_the_state():
    """y' = -r y and 0 = z - c y, with the state as a pytree and the mask matching it."""
    problem = ODEProblem(
        lambda y, p, t: {"y": -p[0] * y["y"], "z": y["z"] - p[1] * y["y"]},
        state={"y": jnp.zeros(()), "z": jnp.zeros(())},
        params=2,
        algebraic={"y": False, "z": True},
    )
    p = jnp.array([0.7, 2.0])
    # the algebraic entry of y0 is only a guess: diffsol projects it onto the constraint
    ys = solve(problem, {"y": jnp.array(1.0), "z": jnp.array(99.0)}, p, T_EVAL, **TOL)
    np.testing.assert_allclose(ys["y"], jnp.exp(-p[0] * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys["z"], p[1] * ys["y"], rtol=1e-6)

    def loss(p):
        y0 = {"y": jnp.array(1.0), "z": jnp.array(0.0)}
        return solve(problem, y0, p, T_EVAL, **TOL)["z"].sum()

    h = 1e-6
    fd = jnp.array(
        [(loss(p.at[i].add(h)) - loss(p.at[i].add(-h))) / (2 * h) for i in range(2)]
    )
    np.testing.assert_allclose(jax.grad(loss)(p), fd, rtol=1e-5)


def test_mass_over_a_pytree_state():
    problem = ODEProblem(
        lambda y, p, t: {"y": -p[0] * y["y"], "z": y["z"] - p[1] * y["y"]},
        state={"y": jnp.zeros(()), "z": jnp.zeros(())},
        params=2,
        mass=lambda v, p, t: {"y": v["y"], "z": jnp.zeros(())},
    )
    p = jnp.array([0.7, 2.0])
    ys = solve(problem, {"y": jnp.array(1.0), "z": jnp.array(2.0)}, p, T_EVAL, **TOL)
    np.testing.assert_allclose(ys["y"], jnp.exp(-p[0] * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys["z"], p[1] * ys["y"], rtol=1e-6)


def test_state_layout_errors():
    with pytest.raises(DiffsolJaxError, match="every leaf of it must be one"):
        ODEProblem(lambda y, p, t: y, state={"pos": jnp.zeros(2), "n": 3}, params=1)
    with pytest.raises(DiffsolJaxError, match="does not match the structure"):
        solve(SPRING, {"pos": POS0}, K, T_EVAL)
    with pytest.raises(DiffsolJaxError, match=r"y0\['pos'\] has the shape \(2,\)"):
        solve(SPRING, {"pos": jnp.zeros(2), "vel": jnp.zeros(N)}, K, T_EVAL)
    with pytest.raises(DiffsolJaxError, match="batch with jax.vmap"):
        solve(SPRING, jax.tree.map(lambda a: jnp.ones((4,) + a.shape), Y0), K, T_EVAL)


def test_rhs_returning_the_wrong_structure_raises():
    problem = ODEProblem(lambda y, p, t: {"pos": y["pos"]}, state=STATE, params=1)
    with pytest.raises(DiffsolJaxError, match="the result of rhs does not match"):
        solve(problem, Y0, K, T_EVAL)
