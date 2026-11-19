"""Tests of the `params` layout of a problem and the values its solves take: one flat array of
parameters, or a pytree whose float-array leaves are the parameters and whose other leaves the
equations capture as constants."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
T0, T1 = T_EVAL[0], T_EVAL[-1]
TOL = {"rtol": 1e-10, "atol": 1e-10}
Y0 = jnp.array([0.1])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Logistic:
    r: jax.Array
    k: jax.Array


LOGISTIC = Logistic(r=jnp.array(1.0), k=jnp.array(1.0))
TREE = ODEProblem(lambda x, p, t: p.r * x * (1.0 - x / p.k), state=1, params=LOGISTIC)
FLAT = ODEProblem(lambda x, p, t: p[0] * x * (1.0 - x / p[1]), state=1, params=2)


def flat_of(v):
    return jnp.array([v.r, v.k])


def tree(y0, v, **kw):
    return solve(TREE, y0, v, T0, T1, saveat=T_EVAL, **{**TOL, **kw})


def flat(y0, p, **kw):
    return solve(FLAT, y0, p, T0, T1, saveat=T_EVAL, **{**TOL, **kw})


def test_solve_matches_flat():
    assert (TREE.nstates, TREE.nparams) == (1, 2)
    ys = tree(Y0, LOGISTIC)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys, flat(Y0, flat_of(LOGISTIC)), rtol=1e-12)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_structure_matches_flat(ad):
    g = jax.grad(lambda v: tree(Y0, v, ad=ad).sum())(LOGISTIC)
    gf = jax.grad(lambda p: flat(Y0, p, ad=ad).sum())(flat_of(LOGISTIC))
    assert isinstance(g, Logistic)
    np.testing.assert_allclose(flat_of(g), gf, rtol=1e-7)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad_of_y0(ad):
    """`y0` is a solve argument, so it is differentiable like the parameters."""
    g = jax.grad(lambda y0: tree(y0, LOGISTIC, ad=ad).sum())(Y0)
    # x(t) = x0 e^t / (1 - x0 + x0 e^t) for r = k = 1
    e = jnp.exp(T_EVAL)
    exact = jax.grad(lambda a: jnp.sum(a * e / (1.0 - a + a * e)))(Y0[0])
    np.testing.assert_allclose(g[0], exact, rtol=1e-6)


def test_y0_as_a_parameter_leaf():
    """The neural-ODE pattern: `y0` is a leaf of the parameters, so its gradient comes back in
    the same structure as the rest of them, ready for optax."""

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class WithY0:
        y0: jax.Array
        rates: Logistic

    v = WithY0(y0=Y0, rates=LOGISTIC)
    problem = ODEProblem(
        lambda x, p, t: p.rates.r * x * (1.0 - x / p.rates.k), state=1, params=v
    )
    assert problem.nparams == 3
    g = jax.grad(lambda v: solve(problem, v.y0, v, T0, T1, saveat=T_EVAL, **TOL).sum())(
        v
    )
    assert isinstance(g, WithY0)
    # the rhs does not use y0, so its gradient comes only from the initial state
    gy = jax.grad(lambda y0: flat(y0, flat_of(LOGISTIC)).sum())(Y0)
    np.testing.assert_allclose(g.y0, gy, rtol=1e-7)
    gf = jax.grad(lambda p: flat(Y0, p).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(flat_of(g.rates), gf, rtol=1e-7)


def test_jacfwd_matches_flat():
    j = jax.jacfwd(lambda v: tree(Y0, v))(LOGISTIC)
    jf = jax.jacfwd(lambda p: flat(Y0, p))(flat_of(LOGISTIC))
    np.testing.assert_allclose(j.r, jf[..., 0], rtol=1e-7)
    np.testing.assert_allclose(j.k, jf[..., 1], rtol=1e-7)


def test_vmap_over_one_leaf():
    rs = jnp.array([0.5, 1.0, 2.0])
    ys = jax.vmap(lambda r: tree(Y0, dataclasses.replace(LOGISTIC, r=r)))(rs)
    loop = jnp.stack([flat(Y0, flat_of(LOGISTIC).at[0].set(r)) for r in rs])
    # the vmap is one batched solve and the loop is three single-lane solves, so they agree to
    # the solver accuracy, not to round-off
    np.testing.assert_allclose(ys, loop, rtol=1e-7)


def test_vmap_over_y0():
    """A `vmap` over `y0` becomes lanes of one batched solve, like a `vmap` over the params."""
    y0s = jnp.array([[0.05], [0.1], [0.4]])
    batched = jax.jit(jax.vmap(tree, in_axes=(0, None)))
    ys = batched(y0s, LOGISTIC)
    loop = jnp.stack([tree(y0, LOGISTIC) for y0 in y0s])
    np.testing.assert_allclose(ys, loop, rtol=1e-8)
    hlo = batched.lower(y0s, LOGISTIC).as_text()
    assert hlo.count("diffsol_solve_dense") == 1


def test_a_template_of_shapes_needs_no_values():
    """Only the shapes and dtypes of the parameter leaves are read, so `jax.eval_shape` of the
    thing being trained is a complete declaration."""
    problem = ODEProblem(
        lambda x, p, t: p.r * x * (1.0 - x / p.k),
        state=1,
        params=jax.eval_shape(lambda: LOGISTIC),
    )
    assert problem.nparams == 2
    np.testing.assert_allclose(
        solve(problem, Y0, LOGISTIC, T0, T1, saveat=T_EVAL, **TOL),
        tree(Y0, LOGISTIC),
        rtol=1e-12,
    )


def test_no_params():
    problem = ODEProblem(lambda x, p, t: x * (1.0 - x), state=1)
    assert problem.nparams == 0
    np.testing.assert_allclose(
        solve(problem, Y0, None, T0, T1, saveat=T_EVAL, **TOL),
        flat(Y0, jnp.ones(2)),
        rtol=1e-9,
    )
    g = jax.grad(
        lambda y0: solve(problem, y0, None, T0, T1, saveat=T_EVAL, **TOL).sum()
    )(Y0)
    gf = jax.grad(lambda y0: flat(y0, jnp.ones(2)).sum())(Y0)
    np.testing.assert_allclose(g, gf, rtol=1e-7)


def test_out_with_tree_params():
    problem = ODEProblem(
        rhs=lambda x, p, t: p.r * x * (1.0 - x / p.k),
        state=1,
        params=LOGISTIC,
        out=lambda x, p, t: jnp.array([2.0 * x[0], p.r * x[0] ** 2]),
    )
    assert problem.nout == 2
    ys = tree(Y0, LOGISTIC)
    outs = solve(problem, Y0, LOGISTIC, T0, T1, saveat=T_EVAL, **TOL)
    np.testing.assert_allclose(outs[:, 0], 2.0 * ys[:, 0], rtol=1e-12)
    np.testing.assert_allclose(outs[:, 1], ys[:, 0] ** 2, rtol=1e-12)
    g = jax.grad(lambda v: solve(problem, Y0, v, T0, T1, saveat=T_EVAL, **TOL).sum())(
        LOGISTIC
    )
    assert np.isfinite(g.r)


def test_dict_params_and_a_callable_leaf():
    """A leaf that is not a float array is a constant of the equations."""
    template = {"r": jnp.array(1.0), "shape": lambda x: x * (1.0 - x)}
    problem = ODEProblem(
        lambda x, p, t: p["r"] * p["shape"](x), state=1, params=template
    )
    assert problem.nparams == 1
    np.testing.assert_allclose(
        solve(problem, Y0, template, T0, T1, saveat=T_EVAL, **TOL),
        flat(Y0, jnp.ones(2)),
        rtol=1e-12,
    )
    g = jax.grad(
        lambda r: solve(
            problem, Y0, {**template, "r": r}, T0, T1, saveat=T_EVAL, **TOL
        ).sum()
    )(template["r"])
    assert np.isfinite(g)


def test_f32_leaf_gets_f32_grad():
    template = dataclasses.replace(LOGISTIC, r=jnp.array(1.0, jnp.float32))
    problem = ODEProblem(
        lambda x, p, t: p.r * x * (1.0 - x / p.k), state=1, params=template
    )
    g = jax.grad(lambda v: solve(problem, Y0, v, T0, T1, saveat=T_EVAL, **TOL).sum())(
        template
    )
    assert g.r.dtype == jnp.float32
    assert g.k.dtype == jnp.float64
    gf = jax.grad(lambda p: flat(Y0, p).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(g.r, gf[0], rtol=1e-5)


def test_integer_leaf_is_captured_as_a_constant():
    """A non-float leaf is compiled into the callbacks, so its value comes from the template."""
    rhs = lambda x, p, t: p["r"] * x * (1.0 - x ** p["n"])  # noqa: E731
    linear = ODEProblem(rhs, state=1, params={"r": jnp.array(1.0), "n": 1})
    cubic = ODEProblem(rhs, state=1, params={"r": jnp.array(1.0), "n": jnp.array(3)})
    assert (linear.nparams, cubic.nparams) == (1, 1)
    args = {"r": jnp.array(1.0), "n": 99}  # the value passed to solve is ignored
    ys_linear = solve(linear, Y0, args, T0, T1, saveat=T_EVAL, **TOL)
    np.testing.assert_allclose(ys_linear, flat(Y0, jnp.ones(2)), rtol=1e-12)
    ys_cubic = solve(cubic, Y0, args, T0, T1, saveat=T_EVAL, **TOL)
    assert np.all(ys_cubic[1:] > ys_linear[1:])


def test_mismatched_params_raise():
    with pytest.raises(DiffsolJaxError, match="does not match the structure"):
        tree(Y0, {"r": jnp.array(1.0), "k": jnp.array(1.0)})
    # an extra axis on a leaf is not lanes: batching is jax.vmap
    with pytest.raises(DiffsolJaxError, match=r"has the shape \(3,\), expected \(\)"):
        tree(Y0, dataclasses.replace(LOGISTIC, r=jnp.ones(3)))


def test_newly_inexact_constant_leaf_raises():
    problem = ODEProblem(
        lambda x, p, t: p["r"] * x * (1.0 - x ** p["n"]),
        state=1,
        params={"r": jnp.array(1.0), "n": 2},
    )
    with pytest.raises(DiffsolJaxError, match="captured it as a constant"):
        solve(
            problem,
            Y0,
            {"r": jnp.array(1.0), "n": jnp.array(2.0)},
            T0,
            T1,
            saveat=T_EVAL,
            **TOL,
        )


def test_a_shape_only_constant_leaf_raises():
    with pytest.raises(DiffsolJaxError, match="has no value"):
        ODEProblem(
            lambda x, p, t: p["r"] * x,
            state=1,
            params={
                "r": jnp.array(1.0),
                "n": jax.ShapeDtypeStruct((), jnp.int32),
            },
        )


def test_dae_with_tree_params():
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class Decay:
        r: jax.Array
        c: jax.Array

    v = Decay(r=jnp.array(0.7), c=jnp.array(2.0))
    y0 = jnp.array([1.0, 0.0])
    problem = ODEProblem(
        lambda x, p, t: jnp.array([-p.r * x[0], p.c * x[0] - x[1]]),
        state=2,
        params=v,
        algebraic=1,
    )
    ys = solve(problem, y0, v, T0, T1, saveat=T_EVAL, **TOL)
    np.testing.assert_allclose(ys[:, 0], jnp.exp(-v.r * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys[:, 1], v.c * ys[:, 0], rtol=1e-6)
    g = jax.grad(
        lambda v: solve(problem, y0, v, T0, T1, saveat=T_EVAL, **TOL)[:, 1].sum()
    )(v)
    np.testing.assert_allclose(g.c, jnp.exp(-v.r * T_EVAL).sum(), rtol=1e-6)


def test_mass_using_a_parameter_leaf_raises():
    with pytest.raises(DiffsolJaxError, match="must not depend on p"):
        ODEProblem(
            lambda x, p, t: jnp.array([-p.r * x[0], p.k * x[0] - x[1]]),
            state=2,
            params=LOGISTIC,
            mass=lambda v, p, t: p.k * jnp.array([v[0], 0.0]),
        )
