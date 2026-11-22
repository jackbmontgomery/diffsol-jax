"""`dtype=float32`: the solver, its callbacks and everything the user sees work in f32, and the
times stay f64."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

F32 = jnp.float32
T_EVAL = jnp.linspace(0.0, 4.0, 5)
T0, T1 = T_EVAL[0], T_EVAL[-1]
# what f32 can hold: tight enough to check against f64, loose enough for the Newton solves
TOL = {"rtol": 1e-5, "atol": 1e-5}


def logistic(y, p, t):
    return p[0] * y * (1.0 - y / p[1])


P32, Y032 = jnp.array([1.0, 2.0], F32), jnp.array([0.1], F32)
LOGISTIC32 = ODEProblem(logistic, state=1, params=2, dtype=F32)
LOGISTIC64 = ODEProblem(logistic, state=1, params=2)
REFERENCE = {"rtol": 1e-12, "atol": 1e-12}


def run(problem, y0, p, **kw):
    return solve(problem, y0, p, T0, T1, saveat=T_EVAL, **kw)


def test_the_solution_matches_f64_to_the_f32_tolerance():
    ys = run(LOGISTIC32, Y032, P32, **TOL)
    ref = run(LOGISTIC64, jnp.array([0.1]), jnp.array([1.0, 2.0]), **REFERENCE)
    assert ys.dtype == F32
    assert np.allclose(np.asarray(ys), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_the_gradient_is_f32_and_matches_f64():
    g = jax.grad(lambda p: run(LOGISTIC32, Y032, p, **TOL).sum())(P32)
    ref = jax.grad(lambda p: run(LOGISTIC64, jnp.array([0.1]), p, **REFERENCE).sum())(
        jnp.array([1.0, 2.0])
    )
    assert g.dtype == F32
    assert np.allclose(np.asarray(g), np.asarray(ref), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_both_ad_modes_work_in_f32(ad):
    g = jax.grad(lambda p: run(LOGISTIC32, Y032, p, ad=ad, **TOL).sum())(P32)
    assert g.dtype == F32
    assert np.all(np.isfinite(np.asarray(g)))


def test_the_two_ad_modes_agree_in_f32():
    adjoint = jax.grad(lambda p: run(LOGISTIC32, Y032, p, **TOL).sum())(P32)
    forward = jax.grad(lambda p: run(LOGISTIC32, Y032, p, ad="forward", **TOL).sum())(
        P32
    )
    assert np.allclose(np.asarray(adjoint), np.asarray(forward), rtol=1e-3, atol=1e-5)


def test_the_gradient_of_y0_is_f32():
    g = jax.grad(lambda y0: run(LOGISTIC32, y0, P32, **TOL).sum())(Y032)
    assert g.dtype == F32
    assert np.all(np.isfinite(np.asarray(g)))


def test_jacfwd_is_f32():
    jac = jax.jacfwd(lambda y0: run(LOGISTIC32, y0, P32, **TOL))(Y032)
    assert jac.dtype == F32
    assert np.all(np.isfinite(np.asarray(jac)))


def test_vmap_is_one_f32_solve():
    y0s = jnp.stack([Y032, Y032 * 2, Y032 * 3])
    batched = jax.jit(jax.vmap(lambda y0: run(LOGISTIC32, y0, P32, **TOL)))
    ys = batched(y0s)
    assert ys.dtype == F32
    assert ys.shape == (3, T_EVAL.shape[0], 1)
    assert batched.lower(y0s).as_text().count("diffsol_solve_dense") == 1
    # the lanes share the step sizes, so a single-lane solve agrees only to the tolerance
    loop = jnp.stack([run(LOGISTIC32, y0, P32, **TOL) for y0 in y0s])
    np.testing.assert_allclose(ys, loop, rtol=1e-3, atol=1e-5)


def test_vmap_of_grad_in_f32():
    ps = jnp.stack([P32, P32 * 1.5])
    got = jax.vmap(jax.grad(lambda p: run(LOGISTIC32, Y032, p, **TOL).sum()))(ps)
    assert got.dtype == F32
    for i, p in enumerate(ps):
        one = jax.grad(lambda p: run(LOGISTIC32, Y032, p, **TOL).sum())(p)
        assert np.allclose(np.asarray(got[i]), np.asarray(one), rtol=1e-3, atol=1e-5)


def test_a_pytree_of_f32_params():
    problem = ODEProblem(
        lambda y, p, t: -p["k"] * y, state=1, params={"k": jnp.ones(1, F32)}, dtype=F32
    )
    g = jax.grad(lambda p: run(problem, jnp.ones(1, F32), p, **TOL).sum())(
        {"k": jnp.ones(1, F32)}
    )
    assert g["k"].dtype == F32
    assert np.all(np.isfinite(np.asarray(g["k"])))


def test_out_returns_f32():
    problem = ODEProblem(
        lambda y, p, t: -p[0] * y,
        state=2,
        params=1,
        out=lambda y, p, t: jnp.stack([y[0] + y[1]]),
        dtype=F32,
    )
    ys = run(problem, jnp.ones(2, F32), jnp.ones(1, F32), **TOL)
    assert ys.dtype == F32
    assert ys.shape == (T_EVAL.shape[0], 1)


def test_a_dae_in_f32():
    """`mass`/`mass_adjoint` take two scalars (`t` and `beta`), so they exercise the f32
    trampoline signature that nothing else does."""
    problem = ODEProblem(
        lambda y, p, t: jnp.stack([-p[0] * y[0], y[0] - y[1]]),
        state=2,
        params=1,
        algebraic=1,
        dtype=F32,
    )
    ys = run(problem, jnp.array([1.0, 1.0], F32), jnp.ones(1, F32), **TOL)
    assert ys.dtype == F32
    # the algebraic row is y1 = y0
    assert np.allclose(np.asarray(ys[:, 0]), np.asarray(ys[:, 1]), atol=1e-5)
    g = jax.grad(lambda p: run(problem, jnp.array([1.0, 1.0], F32), p, **TOL).sum())(
        jnp.ones(1, F32)
    )
    assert g.dtype == F32 and np.all(np.isfinite(np.asarray(g)))


def test_a_stiff_problem_in_f32():
    """Robertson, whose states span 1e-8 to 1: f32 reaches the reference solution to about the
    tolerance it can hold."""
    problem = ODEProblem(
        lambda x, p, t: jnp.stack(
            [
                -p[0] * x[0] + p[2] * x[1] * x[2],
                p[0] * x[0] - p[1] * x[1] ** 2 - p[2] * x[1] * x[2],
                p[1] * x[1] ** 2,
            ]
        ),
        state=3,
        params=3,
        dtype=F32,
    )
    ys = solve(
        problem,
        jnp.array([1.0, 0.0, 0.0], F32),
        jnp.array([0.04, 3e7, 1e4], F32),
        0.0,
        40.0,
        saveat=jnp.array([40.0]),
        rtol=1e-5,
        atol=1e-5,
    )
    expected = np.array([0.7158270687193, 9.185534764529e-6, 0.2841637457471])
    assert np.allclose(np.asarray(ys[0]), expected, rtol=1e-3, atol=1e-6)


def test_f64_inputs_are_cast_to_the_problem_dtype():
    """The problem's dtype wins: a solve takes the values it is given, whatever their dtype."""
    ys = run(LOGISTIC32, jnp.array([0.1]), jnp.array([1.0, 2.0]), **TOL)
    assert ys.dtype == F32
    assert np.allclose(np.asarray(ys), np.asarray(run(LOGISTIC32, Y032, P32, **TOL)))


def test_the_times_stay_f64():
    """`dtype` is the solver's precision, not a JAX x64 switch: diffsol-c takes the times as
    f64, and an f32 time would lose resolution at a large t."""
    ys = solve(
        LOGISTIC32, Y032, P32, 0.0, 4.0, saveat=jnp.array([4.0], jnp.float64), **TOL
    )
    assert ys.dtype == F32


def test_the_default_tolerance_follows_the_dtype():
    """1e-6, the f64 default, is below what f32 can resolve: its adjoint solve diverges. So the
    default must come from the dtype, or `jax.grad` of a default f32 solve would fail."""
    from diffsol_jax._solve import DEFAULT_TOL

    assert DEFAULT_TOL[np.dtype(np.float64)] == 1e-6
    assert DEFAULT_TOL[np.dtype(np.float32)] > 1e-6
    g = jax.grad(lambda p: run(LOGISTIC32, Y032, p).sum())(P32)
    assert g.dtype == F32 and np.all(np.isfinite(np.asarray(g)))


def test_an_f32_and_an_f64_problem_are_independent():
    """Each problem has its own diffsol solver, so the two dtypes coexist in one process."""
    ys32 = run(LOGISTIC32, Y032, P32, **TOL)
    ys64 = run(LOGISTIC64, jnp.array([0.1]), jnp.array([1.0, 2.0]), **REFERENCE)
    assert ys32.dtype == F32 and ys64.dtype == jnp.float64
    assert np.allclose(np.asarray(ys32), np.asarray(ys64), rtol=1e-4, atol=1e-4)


def test_the_dtype_shows_up_in_repr():
    assert "dtype=float32" in repr(LOGISTIC32)
    assert "dtype" not in repr(LOGISTIC64)


def test_replace_keeps_the_dtype():
    other = LOGISTIC32.replace(solver="tsit45")
    assert other.dtype == np.dtype(np.float32)
    assert run(other, Y032, P32, **TOL).dtype == F32


def test_an_unsupported_dtype_is_rejected():
    for bad in (jnp.float16, jnp.int32, "f16", object()):
        with pytest.raises(DiffsolJaxError, match="dtype must be float32 or float64"):
            ODEProblem(logistic, state=1, params=2, dtype=bad)
