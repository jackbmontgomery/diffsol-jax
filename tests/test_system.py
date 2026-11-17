import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, Model, System

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
TOL = {"rtol": 1e-10, "atol": 1e-10}
Y0 = jnp.array([0.1])


class Logistic(System):
    r: jax.Array
    k: jax.Array

    def __call__(self, x, t):
        return self.r * x * (1.0 - x / self.k)


LOGISTIC = Logistic(r=jnp.array(1.0), k=jnp.array(1.0))


def logistic_flat(**kw):
    return Model(
        rhs=lambda x, p, t: p[0] * x * (1.0 - x / p[1]),
        nstates=1,
        nparams=2,
        **TOL,
        **kw,
    )


def flat_of(v):
    return jnp.array([v.r, v.k])


def test_solve_matches_flat():
    model = Model(LOGISTIC, nstates=1, **TOL)
    assert model.nstates == 1 and model.nparams == 2
    ys = model.solve(Y0, LOGISTIC, T_EVAL)
    ys_flat = logistic_flat().solve(Y0, flat_of(LOGISTIC), T_EVAL)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys, ys_flat, rtol=1e-12)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_structure_matches_flat(ad):
    model = Model(LOGISTIC, nstates=1, ad=ad, **TOL)
    flat_model = logistic_flat(ad=ad)
    g = jax.grad(lambda v: model.solve(Y0, v, T_EVAL).sum())(LOGISTIC)
    gf = jax.grad(lambda p: flat_model.solve(Y0, p, T_EVAL).sum())(flat_of(LOGISTIC))
    assert isinstance(g, Logistic)
    np.testing.assert_allclose(flat_of(g), gf, rtol=1e-7)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_of_y0(ad):
    """`y0` is a solve argument, so it is differentiable like the variables."""
    model = Model(LOGISTIC, nstates=1, ad=ad, **TOL)
    g = jax.grad(lambda y0: model.solve(y0, LOGISTIC, T_EVAL).sum())(Y0)
    # x(t) = x0 e^t / (1 - x0 + x0 e^t) for r = k = 1
    e = jnp.exp(T_EVAL)
    exact = jax.grad(lambda a: jnp.sum(a * e / (1.0 - a + a * e)))(Y0[0])
    np.testing.assert_allclose(g[0], exact, rtol=1e-6)


def test_y0_as_a_trainable_field():
    """The neural-ODE pattern: `y0` is a leaf of the variables pytree that is passed to
    `solve`, so its gradient comes back in the same structure as the rest of the variables."""

    class WithY0(Logistic):
        y0: jax.Array

    v = WithY0(r=LOGISTIC.r, k=LOGISTIC.k, y0=Y0)
    model = Model(v, nstates=1, **TOL)
    assert model.nparams == 3
    g = jax.grad(lambda v: model.solve(v.y0, v, T_EVAL).sum())(v)
    assert isinstance(g, WithY0)
    flat = logistic_flat()
    # the rhs does not use y0, so its gradient comes only from the initial state
    gy = jax.grad(lambda y0: flat.solve(y0, flat_of(LOGISTIC), T_EVAL).sum())(Y0)
    np.testing.assert_allclose(g.y0, gy, rtol=1e-7)
    gf = jax.grad(lambda p: flat.solve(Y0, p, T_EVAL).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(flat_of(g), gf, rtol=1e-7)


def test_jacfwd_matches_flat():
    model, flat_model = Model(LOGISTIC, nstates=1, **TOL), logistic_flat()
    j = jax.jacfwd(lambda v: model.solve(Y0, v, T_EVAL))(LOGISTIC)
    jf = jax.jacfwd(lambda p: flat_model.solve(Y0, p, T_EVAL))(flat_of(LOGISTIC))
    np.testing.assert_allclose(j.r, jf[..., 0], rtol=1e-7)
    np.testing.assert_allclose(j.k, jf[..., 1], rtol=1e-7)


def test_vmap_over_one_leaf():
    model, flat_model = Model(LOGISTIC, nstates=1, **TOL), logistic_flat()
    rs = jnp.array([0.5, 1.0, 2.0])
    ys = jax.vmap(lambda r: model.solve(Y0, LOGISTIC.replace(r=r), T_EVAL))(rs)
    flat = jax.vmap(lambda r: flat_of(LOGISTIC).at[0].set(r))(rs)
    np.testing.assert_allclose(ys, flat_model.solve(Y0, flat, T_EVAL), rtol=1e-12)


def test_vmap_over_y0():
    """A `vmap` over `y0` becomes lanes of one batched solve, like a `vmap` over params."""
    model = Model(LOGISTIC, nstates=1, **TOL)
    y0s = jnp.array([[0.05], [0.1], [0.4]])
    batched = jax.jit(jax.vmap(model.solve, in_axes=(0, None, None)))
    ys = batched(y0s, LOGISTIC, T_EVAL)
    loop = jnp.stack([model.solve(y0, LOGISTIC, T_EVAL) for y0 in y0s])
    np.testing.assert_allclose(ys, loop, rtol=1e-8)
    hlo = batched.lower(y0s, LOGISTIC, T_EVAL).as_text()
    assert hlo.count("diffsol_solve_dense") == 1


def test_out_method():
    class WithOut(Logistic):
        def out(self, x, t):
            return jnp.array([2.0 * x[0], x[0] ** 2])

    v = WithOut(r=LOGISTIC.r, k=LOGISTIC.k)
    model = Model(v, nstates=1, **TOL)
    assert model.nout == 2
    ys = Model(LOGISTIC, nstates=1, **TOL).solve(Y0, LOGISTIC, T_EVAL)
    outs = model.solve(Y0, v, T_EVAL)
    np.testing.assert_allclose(outs[:, 0], 2.0 * ys[:, 0], rtol=1e-12)
    np.testing.assert_allclose(outs[:, 1], ys[:, 0] ** 2, rtol=1e-12)
    g = jax.grad(lambda v: model.solve(Y0, v, T_EVAL).sum())(v)
    assert np.isfinite(g.r)


def test_dict_variables_and_static_callable_leaf():
    like = {"r": jnp.array(1.0), "shape": lambda x: x * (1 - x)}
    model = Model(
        rhs=lambda x, v, t: v["r"] * v["shape"](x),
        nstates=1,
        variables_like=like,
        **TOL,
    )
    assert model.nparams == 1
    ys = model.solve(Y0, like, T_EVAL)
    ys_flat = logistic_flat().solve(Y0, jnp.array([1.0, 1.0]), T_EVAL)
    np.testing.assert_allclose(ys, ys_flat, rtol=1e-12)
    g = jax.grad(lambda r: model.solve(Y0, {**like, "r": r}, T_EVAL).sum())(like["r"])
    assert np.isfinite(g)


def test_f32_leaf_gets_f32_grad():
    like = LOGISTIC.replace(r=jnp.array(1.0, jnp.float32))
    model = Model(like, nstates=1, **TOL)
    g = jax.grad(lambda v: model.solve(Y0, v, T_EVAL).sum())(like)
    assert g.r.dtype == jnp.float32
    assert g.k.dtype == jnp.float64
    gf = jax.grad(lambda p: logistic_flat().solve(Y0, p, T_EVAL).sum())(
        flat_of(LOGISTIC)
    )
    np.testing.assert_allclose(g.r, gf[0], rtol=1e-5)


def test_static_metadata_field():
    class WithStatic(System):
        r: jax.Array
        order: int = dataclasses.field(metadata={"static": True})

        def __call__(self, x, t):
            return -self.r * x**self.order

    v = WithStatic(r=jnp.array(1.0), order=3)
    leaves, treedef = jax.tree.flatten(v)
    assert len(leaves) == 1
    assert jax.tree.unflatten(treedef, leaves).order == 3
    ys = Model(v, nstates=1, **TOL).solve(jnp.ones(1), v, T_EVAL)
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

    with pytest.raises(TypeError, match="__call__"):
        Incomplete(r=jnp.array(1.0))


def test_mode_errors():
    def rhs(x, p, t):
        return -x

    with pytest.raises(DiffsolJaxError, match="give a System and nstates"):
        Model(rhs=rhs)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, nstates=1)
    with pytest.raises(DiffsolJaxError, match="exactly one"):
        Model(rhs=rhs, nstates=1, nparams=1, variables_like=LOGISTIC)
    with pytest.raises(DiffsolJaxError, match="no trainable leaves"):
        Model(rhs=rhs, nstates=1, variables_like={"n": 3})
    with pytest.raises(DiffsolJaxError, match="do not also give"):
        Model(LOGISTIC, nstates=1, nparams=2)
    with pytest.raises(DiffsolJaxError, match="give nstates"):
        Model(LOGISTIC)


def test_solve_time_mismatches():
    model = Model(LOGISTIC, nstates=1, **TOL)
    with pytest.raises(DiffsolJaxError, match="does not match"):
        model.solve(Y0, {"r": jnp.array(1.0)}, T_EVAL)
    with pytest.raises(DiffsolJaxError, match="y0 must have the shape"):
        model.solve(jnp.ones(2), LOGISTIC, T_EVAL)


def test_solve_time_newly_inexact_static_leaf():
    like = {"r": jnp.array(1.0), "n": 2}
    model = Model(
        rhs=lambda x, v, t: v["r"] * x * (1.0 - x ** v["n"]),
        nstates=1,
        variables_like=like,
        **TOL,
    )
    with pytest.raises(DiffsolJaxError, match="fixed"):
        model.solve(Y0, {**like, "n": jnp.array(2.0)}, T_EVAL)


class Decay(System):
    """y' = -r y with the algebraic constraint z = c y."""

    r: jax.Array
    c: jax.Array

    def __call__(self, x, t):
        return jnp.array([-self.r * x[0], self.c * x[0] - x[1]])


DECAY = Decay(r=jnp.array(0.7), c=jnp.array(2.0))
DECAY_Y0 = jnp.array([1.0, 0.0])


def check_decay(model, v):
    ys = model.solve(DECAY_Y0, v, T_EVAL)
    np.testing.assert_allclose(ys[:, 0], jnp.exp(-v.r * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys[:, 1], v.c * ys[:, 0], rtol=1e-6)
    g = jax.grad(lambda v: model.solve(DECAY_Y0, v, T_EVAL)[:, 1].sum())(v)
    np.testing.assert_allclose(g.c, jnp.exp(-v.r * T_EVAL).sum(), rtol=1e-6)


def test_dae_with_system_algebraic():
    model = Model(DECAY, nstates=2, algebraic=1, rtol=1e-10, atol=1e-10)
    check_decay(model, DECAY)


def test_dae_with_mass_method():
    class DecayMass(Decay):
        def mass(self, v, t):
            return jnp.array([v[0], 0.0])

    v = DecayMass(r=DECAY.r, c=DECAY.c)
    check_decay(Model(v, nstates=2, rtol=1e-10, atol=1e-10), v)


def test_mass_method_using_trainable_field_raises():
    class BadMass(Decay):
        def mass(self, v, t):
            return self.c * jnp.array([v[0], 0.0])

    with pytest.raises(DiffsolJaxError, match="must not depend on p"):
        Model(BadMass(r=DECAY.r, c=DECAY.c), nstates=2, **TOL)
