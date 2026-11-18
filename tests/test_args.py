"""Tests of the `args` of a solve: one array of parameters, or a pytree whose float-array
leaves are the parameters and whose other leaves the equations capture as constants."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
TOL = {"rtol": 1e-10, "atol": 1e-10}
Y0 = jnp.array([0.1])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Logistic:
    r: jax.Array
    k: jax.Array


LOGISTIC = Logistic(r=jnp.array(1.0), k=jnp.array(1.0))
TREE = ODEProblem(lambda x, p, t: p.r * x * (1.0 - x / p.k))
FLAT = ODEProblem(lambda x, p, t: p[0] * x * (1.0 - x / p[1]))


def flat_of(v):
    return jnp.array([v.r, v.k])


def tree(y0, v, **kw):
    return solve(TREE, y0, T_EVAL, v, **{**TOL, **kw})


def flat(y0, p, **kw):
    return solve(FLAT, y0, T_EVAL, p, **{**TOL, **kw})


def test_solve_matches_flat():
    model = TREE.compile(Y0, LOGISTIC)
    assert (model.nstates, model.nparams) == (1, 2)
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


def test_y0_as_a_leaf_of_args():
    """The neural-ODE pattern: `y0` is a leaf of the `args` pytree, so its gradient comes back
    in the same structure as the rest of the parameters, ready for optax."""

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class WithY0:
        y0: jax.Array
        rates: Logistic

    v = WithY0(y0=Y0, rates=LOGISTIC)
    problem = ODEProblem(lambda x, p, t: p.rates.r * x * (1.0 - x / p.rates.k))
    assert problem.compile(v.y0, v).nparams == 3
    g = jax.grad(lambda v: solve(problem, v.y0, T_EVAL, v, **TOL).sum())(v)
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
    params = jax.vmap(lambda r: flat_of(LOGISTIC).at[0].set(r))(rs)
    np.testing.assert_allclose(ys, flat(Y0, params), rtol=1e-12)


def test_vmap_over_y0():
    """A `vmap` over `y0` becomes lanes of one batched solve, like a `vmap` over the params."""
    y0s = jnp.array([[0.05], [0.1], [0.4]])
    batched = jax.jit(jax.vmap(tree, in_axes=(0, None)))
    ys = batched(y0s, LOGISTIC)
    loop = jnp.stack([tree(y0, LOGISTIC) for y0 in y0s])
    np.testing.assert_allclose(ys, loop, rtol=1e-8)
    hlo = batched.lower(y0s, LOGISTIC).as_text()
    assert hlo.count("diffsol_solve_dense") == 1


def test_no_args():
    problem = ODEProblem(lambda x, p, t: x * (1.0 - x))
    assert problem.compile(Y0).nparams == 0
    np.testing.assert_allclose(
        solve(problem, Y0, T_EVAL, **TOL), flat(Y0, jnp.ones(2)), rtol=1e-9
    )
    g = jax.grad(lambda y0: solve(problem, y0, T_EVAL, **TOL).sum())(Y0)
    gf = jax.grad(lambda y0: flat(y0, jnp.ones(2)).sum())(Y0)
    np.testing.assert_allclose(g, gf, rtol=1e-7)


def test_out_with_tree_args():
    problem = ODEProblem(
        rhs=lambda x, p, t: p.r * x * (1.0 - x / p.k),
        out=lambda x, p, t: jnp.array([2.0 * x[0], p.r * x[0] ** 2]),
    )
    ys = tree(Y0, LOGISTIC)
    outs = solve(problem, Y0, T_EVAL, LOGISTIC, **TOL)
    assert problem.compile(Y0, LOGISTIC).nout == 2
    np.testing.assert_allclose(outs[:, 0], 2.0 * ys[:, 0], rtol=1e-12)
    np.testing.assert_allclose(outs[:, 1], ys[:, 0] ** 2, rtol=1e-12)
    g = jax.grad(lambda v: solve(problem, Y0, T_EVAL, v, **TOL).sum())(LOGISTIC)
    assert np.isfinite(g.r)


def test_dict_args_and_a_callable_leaf():
    """A leaf that is not a float array is a constant of the equations."""
    args = {"r": jnp.array(1.0), "shape": lambda x: x * (1.0 - x)}
    problem = ODEProblem(lambda x, p, t: p["r"] * p["shape"](x))
    assert problem.compile(Y0, args).nparams == 1
    np.testing.assert_allclose(
        solve(problem, Y0, T_EVAL, args, **TOL), flat(Y0, jnp.ones(2)), rtol=1e-12
    )
    g = jax.grad(lambda r: solve(problem, Y0, T_EVAL, {**args, "r": r}, **TOL).sum())(
        args["r"]
    )
    assert np.isfinite(g)


def test_f32_leaf_gets_f32_grad():
    v = dataclasses.replace(LOGISTIC, r=jnp.array(1.0, jnp.float32))
    g = jax.grad(lambda v: tree(Y0, v).sum())(v)
    assert g.r.dtype == jnp.float32
    assert g.k.dtype == jnp.float64
    gf = jax.grad(lambda p: flat(Y0, p).sum())(flat_of(LOGISTIC))
    np.testing.assert_allclose(g.r, gf[0], rtol=1e-5)


def test_integer_leaf_is_captured_as_a_constant():
    problem = ODEProblem(lambda x, p, t: p["r"] * x * (1.0 - x ** p["n"]))
    args = {"r": jnp.array(1.0), "n": 1}
    assert problem.compile(Y0, args).nparams == 1
    linear = solve(problem, Y0, T_EVAL, args, **TOL)
    np.testing.assert_allclose(linear, flat(Y0, jnp.ones(2)), rtol=1e-12)
    # a different constant is a different solver, with the value compiled into its callbacks
    cubic = solve(problem, Y0, T_EVAL, {**args, "n": 3}, **TOL)
    assert problem.compile(Y0, args) is not problem.compile(Y0, {**args, "n": 3})
    assert np.all(cubic[1:] > linear[1:])
    # the same, as an int array
    np.testing.assert_allclose(
        solve(problem, Y0, T_EVAL, {**args, "n": jnp.array(3)}, **TOL), cubic, rtol=1e-9
    )


def test_a_traced_constant_leaf_raises():
    problem = ODEProblem(lambda x, p, t: p["r"] * x * (1.0 - x ** p["n"]))

    def f(args):
        return solve(problem, Y0, T_EVAL, args, **TOL)

    args = {"r": jnp.array(1.0), "n": jnp.array(3)}
    with pytest.raises(DiffsolJaxError, match="not differentiable"):
        jax.jit(f)(args)


def test_a_new_structure_builds_a_new_solver():
    """Changing the structure or the shapes of `args` is not an error: it builds (and compiles)
    another solver, the way jax.jit traces again for a new shape."""
    first = TREE.compile(Y0, LOGISTIC)
    wider = dataclasses.replace(LOGISTIC, r=jnp.ones(3))
    assert TREE.compile(Y0, wider) is not first
    assert TREE.compile(Y0, LOGISTIC) is first
    problem = ODEProblem(lambda x, p, t: p["r"] * x * (1.0 - x / p["k"]))
    np.testing.assert_allclose(
        solve(problem, Y0, T_EVAL, {"r": jnp.array(1.0), "k": jnp.array(1.0)}, **TOL),
        tree(Y0, LOGISTIC),
        rtol=1e-12,
    )


def test_dae_with_tree_args():
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class Decay:
        r: jax.Array
        c: jax.Array

    v = Decay(r=jnp.array(0.7), c=jnp.array(2.0))
    y0 = jnp.array([1.0, 0.0])
    problem = ODEProblem(
        lambda x, p, t: jnp.array([-p.r * x[0], p.c * x[0] - x[1]]), algebraic=1
    )
    ys = solve(problem, y0, T_EVAL, v, **TOL)
    np.testing.assert_allclose(ys[:, 0], jnp.exp(-v.r * T_EVAL), rtol=1e-6)
    np.testing.assert_allclose(ys[:, 1], v.c * ys[:, 0], rtol=1e-6)
    g = jax.grad(lambda v: solve(problem, y0, T_EVAL, v, **TOL)[:, 1].sum())(v)
    np.testing.assert_allclose(g.c, jnp.exp(-v.r * T_EVAL).sum(), rtol=1e-6)


def test_mass_using_a_parameter_leaf_raises():
    problem = ODEProblem(
        lambda x, p, t: jnp.array([-p.r * x[0], p.k * x[0] - x[1]]),
        mass=lambda v, p, t: p.k * jnp.array([v[0], 0.0]),
    )
    with pytest.raises(DiffsolJaxError, match="must not depend on p"):
        solve(problem, jnp.array([1.0, 0.0]), T_EVAL, LOGISTIC)
