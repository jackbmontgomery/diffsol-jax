"""Tests of pytree variables: `Model(variables_like=...)` and the `Variables` base class."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from diffsol_jax import DiffsolJaxError, Model, Variables

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
TOL = {"rtol": 1e-10, "atol": 1e-10}


class Logistic(Variables):
    x0: jax.Array
    r: jax.Array
    k: jax.Array


LOGISTIC = Logistic(x0=jnp.array([0.1]), r=jnp.array(1.0), k=jnp.array(1.0))


def logistic_pytree(like=LOGISTIC, **kw):
    return Model(
        rhs=lambda x, v, t: v.r * x * (1.0 - x / v.k),
        init=lambda v, t: v.x0,
        nstates=1,
        variables_like=like,
        **TOL,
        **kw,
    )


def logistic_flat(**kw):
    return Model(
        rhs=lambda x, p, t: p[1] * x * (1.0 - x / p[2]),
        init=lambda p, t: p[:1],
        nstates=1,
        nparams=3,
        **TOL,
        **kw,
    )


def flat_of(v):
    return jnp.array([v.x0[0], v.r, v.k])


def test_solve_matches_flat():
    ys = logistic_pytree().solve(LOGISTIC, T_EVAL)
    ys_flat = logistic_flat().solve(flat_of(LOGISTIC), T_EVAL)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys, ys_flat, rtol=1e-12)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_structure_matches_flat(ad):
    model, flat_model = logistic_pytree(ad=ad), logistic_flat(ad=ad)
    g = jax.grad(lambda v: model.solve(v, T_EVAL).sum())(LOGISTIC)
    gf = jax.grad(lambda p: flat_model.solve(p, T_EVAL).sum())(flat_of(LOGISTIC))
    assert isinstance(g, Logistic)
    np.testing.assert_allclose(flat_of(g), gf, rtol=1e-7)


def test_jacfwd_matches_flat():
    model, flat_model = logistic_pytree(), logistic_flat()
    j = jax.jacfwd(lambda v: model.solve(v, T_EVAL))(LOGISTIC)
    jf = jax.jacfwd(lambda p: flat_model.solve(p, T_EVAL))(flat_of(LOGISTIC))
    np.testing.assert_allclose(j.r, jf[..., 1], rtol=1e-7)
    np.testing.assert_allclose(j.x0[..., 0], jf[..., 0], rtol=1e-7)


def test_vmap_over_one_leaf():
    model, flat_model = logistic_pytree(), logistic_flat()
    rs = jnp.array([0.5, 1.0, 2.0])
    ys = jax.vmap(lambda r: model.solve(LOGISTIC.replace(r=r), T_EVAL))(rs)
    flat = jax.vmap(lambda r: flat_of(LOGISTIC).at[1].set(r))(rs)
    np.testing.assert_allclose(ys, flat_model.solve(flat, T_EVAL), rtol=1e-12)


def test_dict_variables_and_static_callable_leaf():
    like = {"r": jnp.array(1.0), "x0": jnp.array([0.1]), "shape": lambda x: x * (1 - x)}
    model = Model(
        rhs=lambda x, v, t: v["r"] * v["shape"](x),
        init=lambda v, t: v["x0"],
        nstates=1,
        variables_like=like,
        **TOL,
    )
    assert model.nparams == 2
    ys = model.solve(like, T_EVAL)
    ys_flat = logistic_flat().solve(jnp.array([0.1, 1.0, 1.0]), T_EVAL)
    np.testing.assert_allclose(ys, ys_flat, rtol=1e-12)
    g = jax.grad(lambda r: model.solve({**like, "r": r}, T_EVAL).sum())(like["r"])
    assert np.isfinite(g)


def test_f32_leaf_gets_f32_grad():
    like = LOGISTIC.replace(r=jnp.array(1.0, jnp.float32))
    model = logistic_pytree(like)
    g = jax.grad(lambda v: model.solve(v, T_EVAL).sum())(like)
    assert g.r.dtype == jnp.float32
    assert g.k.dtype == jnp.float64
    gf = jax.grad(lambda p: logistic_flat().solve(p, T_EVAL).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(g.r, gf[1], rtol=1e-5)


def test_static_metadata_field():
    import dataclasses

    class WithStatic(Variables):
        r: jax.Array
        order: int = dataclasses.field(metadata={"static": True})

    v = WithStatic(r=jnp.array(1.0), order=3)
    leaves, treedef = jax.tree.flatten(v)
    assert len(leaves) == 1
    assert jax.tree.unflatten(treedef, leaves).order == 3


def test_replace_is_functional():
    v = LOGISTIC.replace(r=jnp.array(2.0))
    assert v.r == 2.0 and LOGISTIC.r == 1.0
    with pytest.raises(Exception):
        LOGISTIC.r = jnp.array(3.0)


def test_mode_errors():
    rhs, init = lambda x, p, t: -x, lambda p, t: jnp.ones(1)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, init=init, nstates=1)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, init=init, nstates=1, nparams=1, variables_like=LOGISTIC)
    with pytest.raises(DiffsolJaxError, match="no trainable leaves"):
        Model(rhs=rhs, init=init, nstates=1, variables_like={"n": 3})


def test_solve_time_mismatches():
    model = logistic_pytree()
    with pytest.raises(DiffsolJaxError, match="does not match"):
        model.solve({"r": jnp.array(1.0)}, T_EVAL)
    with pytest.raises(DiffsolJaxError, match="shape"):
        model.solve(LOGISTIC.replace(x0=jnp.ones(2)), T_EVAL)


def test_solve_time_newly_inexact_static_leaf():
    like = {"r": jnp.array(1.0), "x0": jnp.array([0.1]), "n": 2}
    model = Model(
        rhs=lambda x, v, t: v["r"] * x * (1.0 - x**v["n"]),
        init=lambda v, t: v["x0"],
        nstates=1,
        variables_like=like,
        **TOL,
    )
    with pytest.raises(DiffsolJaxError, match="fixed"):
        model.solve({**like, "n": jnp.array(2.0)}, T_EVAL)


def test_dae_with_pytree_variables():
    """y' = -r y with the algebraic constraint z = c y, as a mass-matrix DAE."""

    class Dae(Variables):
        r: jax.Array
        c: jax.Array

    v0 = Dae(r=jnp.array(0.7), c=jnp.array(2.0))

    def rhs(x, v, t):
        return jnp.array([-v.r * x[0], v.c * x[0] - x[1]])

    model = Model(
        rhs=rhs,
        init=lambda v, t: jnp.array([1.0, 0.0]),
        nstates=2,
        variables_like=v0,
        algebraic=1,
        rtol=1e-10,
        atol=1e-10,
    )
    ys = model.solve(v0, T_EVAL)
    np.testing.assert_allclose(ys[:, 0], jnp.exp(-v0.r * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys[:, 1], v0.c * ys[:, 0], rtol=1e-6)
    g = jax.grad(lambda v: model.solve(v, T_EVAL)[:, 1].sum())(v0)
    expect_c = jnp.exp(-v0.r * T_EVAL).sum()
    np.testing.assert_allclose(g.c, expect_c, rtol=1e-6)
