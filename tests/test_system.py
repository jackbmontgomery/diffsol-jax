"""Tests of `System` (`Model(system)`) and the lower-level `Model(variables_like=...)`."""

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from diffsol_jax import DiffsolJaxError, Model, System

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
TOL = {"rtol": 1e-10, "atol": 1e-10}


class Logistic(System):
    x0: jax.Array
    r: jax.Array
    k: jax.Array

    def init(self, t):
        return self.x0

    def __call__(self, x, t):
        return self.r * x * (1.0 - x / self.k)


LOGISTIC = Logistic(x0=jnp.array([0.1]), r=jnp.array(1.0), k=jnp.array(1.0))


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
    model = Model(LOGISTIC, **TOL)
    assert model.nstates == 1 and model.nparams == 3
    ys = model.solve(LOGISTIC, T_EVAL)
    ys_flat = logistic_flat().solve(flat_of(LOGISTIC), T_EVAL)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys, ys_flat, rtol=1e-12)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_structure_matches_flat(ad):
    model, flat_model = Model(LOGISTIC, ad=ad, **TOL), logistic_flat(ad=ad)
    g = jax.grad(lambda v: model.solve(v, T_EVAL).sum())(LOGISTIC)
    gf = jax.grad(lambda p: flat_model.solve(p, T_EVAL).sum())(flat_of(LOGISTIC))
    assert isinstance(g, Logistic)
    np.testing.assert_allclose(flat_of(g), gf, rtol=1e-7)


def test_jacfwd_matches_flat():
    model, flat_model = Model(LOGISTIC, **TOL), logistic_flat()
    j = jax.jacfwd(lambda v: model.solve(v, T_EVAL))(LOGISTIC)
    jf = jax.jacfwd(lambda p: flat_model.solve(p, T_EVAL))(flat_of(LOGISTIC))
    np.testing.assert_allclose(j.r, jf[..., 1], rtol=1e-7)
    np.testing.assert_allclose(j.x0[..., 0], jf[..., 0], rtol=1e-7)


def test_vmap_over_one_leaf():
    model, flat_model = Model(LOGISTIC, **TOL), logistic_flat()
    rs = jnp.array([0.5, 1.0, 2.0])
    ys = jax.vmap(lambda r: model.solve(LOGISTIC.replace(r=r), T_EVAL))(rs)
    flat = jax.vmap(lambda r: flat_of(LOGISTIC).at[1].set(r))(rs)
    np.testing.assert_allclose(ys, flat_model.solve(flat, T_EVAL), rtol=1e-12)


def test_out_method():
    class WithOut(Logistic):
        def out(self, x, t):
            return jnp.array([2.0 * x[0], x[0] ** 2])

    v = WithOut(x0=LOGISTIC.x0, r=LOGISTIC.r, k=LOGISTIC.k)
    model = Model(v, **TOL)
    assert model.nout == 2
    ys = Model(LOGISTIC, **TOL).solve(LOGISTIC, T_EVAL)
    outs = model.solve(v, T_EVAL)
    np.testing.assert_allclose(outs[:, 0], 2.0 * ys[:, 0], rtol=1e-12)
    np.testing.assert_allclose(outs[:, 1], ys[:, 0] ** 2, rtol=1e-12)
    g = jax.grad(lambda v: model.solve(v, T_EVAL).sum())(v)
    assert np.isfinite(g.r)


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
    model = Model(like, **TOL)
    g = jax.grad(lambda v: model.solve(v, T_EVAL).sum())(like)
    assert g.r.dtype == jnp.float32
    assert g.k.dtype == jnp.float64
    gf = jax.grad(lambda p: logistic_flat().solve(p, T_EVAL).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(g.r, gf[1], rtol=1e-5)


def test_static_metadata_field():
    class WithStatic(System):
        r: jax.Array
        order: int = dataclasses.field(metadata={"static": True})

        def init(self, t):
            return jnp.ones(1)

        def __call__(self, x, t):
            return -self.r * x**self.order

    v = WithStatic(r=jnp.array(1.0), order=3)
    leaves, treedef = jax.tree.flatten(v)
    assert len(leaves) == 1
    assert jax.tree.unflatten(treedef, leaves).order == 3
    ys = Model(v, **TOL).solve(v, T_EVAL)
    exact = (1.0 + 2.0 * T_EVAL) ** -0.5  # x' = -x^3, x(0) = 1
    np.testing.assert_allclose(ys[:, 0], exact, rtol=1e-7)


def test_replace_is_functional():
    v = LOGISTIC.replace(r=jnp.array(2.0))
    assert v.r == 2.0 and LOGISTIC.r == 1.0
    with pytest.raises(Exception):
        LOGISTIC.r = jnp.array(3.0)


def test_abstract_methods_enforced():
    class Incomplete(System):
        r: jax.Array

        def init(self, t):
            return jnp.ones(1)

    with pytest.raises(TypeError, match="__call__"):
        Incomplete(r=jnp.array(1.0))


def test_mode_errors():
    rhs, init = lambda x, p, t: -x, lambda p, t: jnp.ones(1)
    with pytest.raises(DiffsolJaxError, match="give a System"):
        Model(rhs=rhs, init=init)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, init=init, nstates=1)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, init=init, nstates=1, nparams=1, variables_like=LOGISTIC)
    with pytest.raises(DiffsolJaxError, match="no trainable leaves"):
        Model(rhs=rhs, init=init, nstates=1, variables_like={"n": 3})
    with pytest.raises(DiffsolJaxError, match="do not also give"):
        Model(LOGISTIC, nstates=1)


def test_solve_time_mismatches():
    model = Model(LOGISTIC, **TOL)
    with pytest.raises(DiffsolJaxError, match="does not match"):
        model.solve({"r": jnp.array(1.0)}, T_EVAL)
    with pytest.raises(DiffsolJaxError, match="shape"):
        model.solve(LOGISTIC.replace(x0=jnp.ones(2)), T_EVAL)


def test_solve_time_newly_inexact_static_leaf():
    like = {"r": jnp.array(1.0), "x0": jnp.array([0.1]), "n": 2}
    model = Model(
        rhs=lambda x, v, t: v["r"] * x * (1.0 - x ** v["n"]),
        init=lambda v, t: v["x0"],
        nstates=1,
        variables_like=like,
        **TOL,
    )
    with pytest.raises(DiffsolJaxError, match="fixed"):
        model.solve({**like, "n": jnp.array(2.0)}, T_EVAL)


class Decay(System):
    """y' = -r y with the algebraic constraint z = c y."""

    r: jax.Array
    c: jax.Array

    def init(self, t):
        return jnp.array([1.0, 0.0])

    def __call__(self, x, t):
        return jnp.array([-self.r * x[0], self.c * x[0] - x[1]])


DECAY = Decay(r=jnp.array(0.7), c=jnp.array(2.0))


def check_decay(model, v):
    ys = model.solve(v, T_EVAL)
    np.testing.assert_allclose(ys[:, 0], jnp.exp(-v.r * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys[:, 1], v.c * ys[:, 0], rtol=1e-6)
    g = jax.grad(lambda v: model.solve(v, T_EVAL)[:, 1].sum())(v)
    np.testing.assert_allclose(g.c, jnp.exp(-v.r * T_EVAL).sum(), rtol=1e-6)


def test_dae_with_system_algebraic():
    check_decay(Model(DECAY, algebraic=1, rtol=1e-10, atol=1e-10), DECAY)


def test_dae_with_mass_method():
    class DecayMass(Decay):
        def mass(self, v, t):
            return jnp.array([v[0], 0.0])

    v = DecayMass(r=DECAY.r, c=DECAY.c)
    check_decay(Model(v, rtol=1e-10, atol=1e-10), v)


def test_mass_method_using_trainable_field_raises():
    class BadMass(Decay):
        def mass(self, v, t):
            return self.c * jnp.array([v[0], 0.0])

    with pytest.raises(DiffsolJaxError, match="must not depend on p"):
        Model(BadMass(r=DECAY.r, c=DECAY.c), **TOL)
