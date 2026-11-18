import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
X0 = 0.1
Y0 = jnp.array([X0])
TOL = {"rtol": 1e-10, "atol": 1e-10}

LOGISTIC = ODEProblem(lambda x, p, t: p[0] * x * (1.0 - x / p[1]))


def logistic(y0, p, t_eval=T_EVAL, **kw):
    return solve(LOGISTIC, y0, t_eval, p, **{**TOL, **kw})


def logistic_exact(r, t):
    return X0 * jnp.exp(r * t) / (1.0 - X0 + X0 * jnp.exp(r * t))


def robertson_rhs(x, p, t):
    k1, k2, k3 = p
    return jnp.array(
        [
            -k1 * x[0] + k3 * x[1] * x[2],
            k1 * x[0] - k2 * x[1] ** 2 - k3 * x[1] * x[2],
            k2 * x[1] ** 2,
        ]
    )


ROBERTSON = ODEProblem(robertson_rhs)
ROBERTSON_TOL = {"rtol": 1e-8, "atol": 1e-10}
ROBERTSON_P = jnp.array([0.04, 3e7, 1e4])
Y0_ROB = jnp.array([1.0, 0.0, 0.0])
# the classic reference solution at t = 40
ROBERTSON_40 = np.array([0.7158270687193, 9.185534764529e-6, 0.2841637457471])


def robertson(y0, p, t_eval, **kw):
    return solve(ROBERTSON, y0, t_eval, p, **{**ROBERTSON_TOL, **kw})


def test_logistic_solve():
    ys = logistic(Y0, jnp.array([1.0, 1.0]))
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys[:, 0], logistic_exact(1.0, T_EVAL), rtol=1e-7)


def test_logistic_out():
    problem = ODEProblem(
        rhs=lambda x, p, t: p[0] * x * (1.0 - x),
        out=lambda x, p, t: jnp.array([2.0 * x[0], x[0] ** 2]),
    )
    ys = solve(problem, Y0, T_EVAL, jnp.array([1.0]), **TOL)
    x = logistic_exact(1.0, T_EVAL)
    np.testing.assert_allclose(ys, jnp.stack([2 * x, x**2], axis=-1), rtol=1e-7)


def test_nout_is_inferred():
    problem = ODEProblem(
        rhs=lambda x, p, t: -x, out=lambda x, p, t: jnp.concatenate([x, x, x])
    )
    model = problem.compile(Y0, jnp.array([1.0]))
    assert (model.nstates, model.nparams, model.nout) == (1, 1, 3)
    assert solve(problem, Y0, T_EVAL, jnp.array([1.0])).shape == (5, 3)


@pytest.mark.parametrize("solver", ["bdf", "esdirk34", "tr_bdf2"])
def test_robertson(solver):
    ys = jax.jit(lambda y0, p: robertson(y0, p, jnp.array([0.0, 40.0]), solver=solver))(
        Y0_ROB, ROBERTSON_P
    )
    np.testing.assert_allclose(ys[-1], ROBERTSON_40, rtol=1e-4)


def test_vmap_is_one_solve_and_matches_a_loop():
    params = jnp.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.5]])
    batched = jax.jit(jax.vmap(logistic, in_axes=(None, 0)))
    ys = batched(Y0, params)
    loop = jnp.stack([logistic(Y0, p) for p in params])
    np.testing.assert_allclose(ys, loop, rtol=1e-8)
    hlo = batched.lower(Y0, params).as_text()
    assert hlo.count("diffsol_solve_dense") == 1


def loss(p, **kw):
    return jnp.sum(logistic(Y0, p, **kw) ** 2)


def finite_difference(f, p, h=1e-6):
    return jnp.array(
        [(f(p.at[i].add(h)) - f(p.at[i].add(-h))) / (2 * h) for i in range(p.shape[0])]
    )


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad(ad):
    p = jnp.array([1.2, 1.5])
    grad = jax.jit(jax.grad(lambda p: loss(p, ad=ad)))(p)
    np.testing.assert_allclose(grad, finite_difference(loss, p), rtol=1e-5)


def test_forward_and_adjoint_jacobians_agree():
    p = jnp.array([1.2, 1.5])
    fwd = jax.jacfwd(lambda p: logistic(Y0, p, ad="forward"))(p)
    # jacrev maps over the cotangent: one backward solve with nout * nt directions
    rev = jax.jit(jax.jacrev(lambda p: logistic(Y0, p, ad="adjoint")))(p)
    np.testing.assert_allclose(rev, fwd, rtol=1e-6, atol=1e-10)


def test_jacobians_with_either_ad_mode():
    # one primitive carries both AD rules: jacfwd works with ad="adjoint" (the forward
    # sensitivities), and jacrev with ad="forward" (their transpose)
    p = jnp.array([1.2, 1.5])
    fwd = jax.jacfwd(lambda p: logistic(Y0, p, ad="adjoint"))(p)
    rev = jax.jacrev(lambda p: logistic(Y0, p, ad="forward"))(p)
    np.testing.assert_allclose(rev, fwd, rtol=1e-6, atol=1e-10)


def test_vmap_of_jacrev():
    # the vmapped lane axis and the jacrev direction axis go through one backward solve
    params = jnp.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.5]])

    def f(p):
        return logistic(Y0, p)

    jac = jax.jit(jax.vmap(jax.jacrev(f)))(params)
    loop = jnp.stack([jax.jacrev(f)(p) for p in params])
    np.testing.assert_allclose(jac, loop, rtol=1e-6, atol=1e-10)


def test_vmap_of_grad():
    params = jnp.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.5]])
    grads = jax.jit(jax.vmap(jax.grad(loss)))(params)
    loop = jnp.stack([jax.grad(loss)(p) for p in params])
    # a batched solve takes one step sequence for all the lanes, and the backward pass uses the
    # default adjoint tolerances: the lanes agree to the solver accuracy, not to round-off
    np.testing.assert_allclose(grads, loop, rtol=1e-6)


def test_robertson_grad():
    t = jnp.array([0.0, 1.0, 10.0, 40.0])

    def f(logp):
        return jnp.sum(robertson(Y0_ROB, jnp.exp(logp), t)[:, 0])

    logp = jnp.log(ROBERTSON_P)
    grad = jax.jit(jax.grad(f))(logp)
    np.testing.assert_allclose(grad, finite_difference(f, logp, 1e-5), rtol=1e-3)


def test_vjp_function_can_run_two_times():
    ys, vjp = jax.vjp(lambda p: logistic(Y0, p), jnp.array([1.0, 1.0]))
    (g1,) = vjp(jnp.ones_like(ys))
    (g2,) = vjp(jnp.ones_like(ys))
    np.testing.assert_allclose(g1, g2)


def test_solves_of_one_problem_share_a_solver():
    """The solver (and so its compiled callbacks and its checkpoints) is cached on the problem,
    the shapes and the compile-time options; the solve-time options do not build a new one."""
    p = jnp.array([1.0, 1.0])
    model = LOGISTIC.compile(Y0, p)
    assert LOGISTIC.compile(Y0, p) is model
    assert LOGISTIC.compile(Y0, p, solver="tsit45") is not model
    logistic(Y0, p)
    logistic(Y0, p, rtol=1e-8, ad="forward")
    assert LOGISTIC.compile(Y0, p) is model
    # a different state size, parameter layout or compile-time option is a different solver
    assert LOGISTIC.compile(jnp.ones(2), p) is not model
    assert LOGISTIC.compile(Y0, {"r": p[0], "k": p[1]}) is not model
    assert LOGISTIC.compile(Y0, p, max_checkpoints=1) is not model


def test_evicted_checkpoint_is_recomputed():
    p = jnp.array([1.0, 1.0])
    model = LOGISTIC.compile(Y0, p, max_checkpoints=1)
    ys, vjp = jax.vjp(lambda p: logistic(Y0, p, max_checkpoints=1), p)
    (expected,) = vjp(jnp.ones_like(ys))
    before = model.num_recomputes()
    # a second forward solve frees the first checkpoint
    jax.vjp(lambda p: logistic(Y0, p, max_checkpoints=1), jnp.array([2.0, 1.0]))
    (grad,) = vjp(jnp.ones_like(ys))
    assert model.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_grad_of_scan_with_more_solves_than_checkpoints():
    """`jax.grad` of a scan runs all the forward solves before the backward solves, so most of
    the checkpoints are freed; the backward solves run the forward solves again."""
    model = LOGISTIC.compile(Y0, jnp.zeros(2), max_checkpoints=2)

    def f(p):
        def body(acc, r):
            return acc + jnp.sum(logistic(Y0, p * r, max_checkpoints=2) ** 2), None

        return jax.lax.scan(body, 0.0, jnp.linspace(0.8, 1.2, 6))[0]

    p = jnp.array([1.0, 1.5])
    before = model.num_recomputes()
    grad = jax.jit(jax.grad(f))(p)
    np.testing.assert_allclose(grad, finite_difference(f, p), rtol=1e-5)
    assert model.num_recomputes() > before


def _bkwd(model, handle, params, ct):
    """Call the backward primitive directly, as a wrong residual would reach it."""
    from diffsol_jax import _primitives

    kw = {"model": model, **TOL}
    prim = _primitives.VjpBkwd(
        jax.typeof(handle), jax.typeof(params), jax.typeof(T_EVAL), jax.typeof(ct), **kw
    )
    return prim(handle, params, T_EVAL, ct)


def _fwd(model, params):
    """`params` here is the internal vector `[p0, p1, y0]` that the primitives take."""
    from diffsol_jax import _primitives

    kw = {"model": model, **TOL}
    prim = _primitives._VjpFwdOp(jax.typeof(params), jax.typeof(T_EVAL), **kw)
    return prim(params, T_EVAL)


def test_handle_with_other_params_is_not_used():
    model = LOGISTIC.compile(Y0, jnp.zeros(2))
    p1, p2 = jnp.array([[1.0, 1.0, X0]]), jnp.array([[2.0, 1.5, X0]])
    ys, h1 = _fwd(model, p1)
    ct = jnp.ones_like(ys)
    expected = _bkwd(model, _fwd(model, p2)[1], p2, ct)
    before = model.num_recomputes()
    grad = _bkwd(model, h1, p2, ct)  # the handle of p1 with the params p2
    assert model.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_mixed_handles_are_not_used():
    model = LOGISTIC.compile(Y0, jnp.zeros(2))
    p = jnp.array([[1.0, 1.0, X0], [2.0, 1.5, X0]])
    ys, h = _fwd(model, p)
    ct = jnp.ones_like(ys)
    expected = _bkwd(model, h, p, ct)
    # lane 1 has the id of a different forward solve
    _, other = _fwd(model, p[1:])
    before = model.num_recomputes()
    grad = _bkwd(model, h.at[1].set(other[0]), p, ct)
    assert model.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_mismatched_batch_axes_raise():
    with pytest.raises(DiffsolJaxError, match="do not broadcast together"):
        solve(LOGISTIC, jnp.ones((3, 1)), T_EVAL, jnp.ones((4, 2)))


def test_argument_errors():
    p = jnp.array([1.0, 1.0])
    with pytest.raises(DiffsolJaxError, match="t_eval must have one axis"):
        solve(LOGISTIC, Y0, jnp.ones((2, 2)), p)
    with pytest.raises(DiffsolJaxError, match="y0 must have the shape"):
        solve(LOGISTIC, jnp.float64(0.1), T_EVAL, p)
    with pytest.raises(DiffsolJaxError, match="ad must be"):
        solve(LOGISTIC, Y0, T_EVAL, p, ad="reverse")
    with pytest.raises(DiffsolJaxError, match="solver must be one of"):
        solve(LOGISTIC, Y0, T_EVAL, p, solver="rk4")
    with pytest.raises(DiffsolJaxError, match="rhs must be a function"):
        ODEProblem(rhs=None)


def test_model_is_not_the_entry_point():
    from diffsol_jax import Model

    with pytest.raises(DiffsolJaxError, match="no longer the entry point"):
        Model(rhs=lambda x, p, t: -x, nstates=1, nparams=1)
    with pytest.raises(DiffsolJaxError, match="no longer the entry point"):
        Model(lambda x, p, t: -x)
