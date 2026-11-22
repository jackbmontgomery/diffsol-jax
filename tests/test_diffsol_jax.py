import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import DiffsolJaxError, ODEProblem, solve

jax.config.update("jax_enable_x64", True)

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
T0, T1 = T_EVAL[0], T_EVAL[-1]
X0 = 0.1
Y0 = jnp.array([X0])
TOL = {"rtol": 1e-10, "atol": 1e-10}
# the primitives take every solve option explicitly; the two helpers below must pass the same
# ones, or the backward solve would miss the checkpoint of the forward solve
PRIM_KW = {**TOL, "max_steps": 10_000}

LOGISTIC = ODEProblem(lambda x, p, t: p[0] * x * (1.0 - x / p[1]), state=1, params=2)


def logistic(y0, p, saveat=T_EVAL, **kw):
    return solve(LOGISTIC, y0, p, saveat[0], saveat[-1], saveat=saveat, **{**TOL, **kw})


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


ROBERTSON = ODEProblem(robertson_rhs, state=3, params=3)
ROBERTSON_TOL = {"rtol": 1e-8, "atol": 1e-10}
ROBERTSON_P = jnp.array([0.04, 3e7, 1e4])
Y0_ROB = jnp.array([1.0, 0.0, 0.0])
# the classic reference solution at t = 40
ROBERTSON_40 = np.array([0.7158270687193, 9.185534764529e-6, 0.2841637457471])


def robertson(y0, p, saveat, problem=ROBERTSON, **kw):
    return solve(
        problem, y0, p, saveat[0], saveat[-1], saveat=saveat, **{**ROBERTSON_TOL, **kw}
    )


def test_logistic_solve():
    ys = logistic(Y0, jnp.array([1.0, 1.0]))
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys[:, 0], logistic_exact(1.0, T_EVAL), rtol=1e-7)


def test_logistic_out():
    problem = ODEProblem(
        rhs=lambda x, p, t: p[0] * x * (1.0 - x),
        state=1,
        params=1,
        out=lambda x, p, t: jnp.array([2.0 * x[0], x[0] ** 2]),
    )
    ys = solve(problem, Y0, jnp.array([1.0]), T0, T1, saveat=T_EVAL, **TOL)
    x = logistic_exact(1.0, T_EVAL)
    np.testing.assert_allclose(ys, jnp.stack([2 * x, x**2], axis=-1), rtol=1e-7)


def test_nout_is_inferred():
    problem = ODEProblem(
        rhs=lambda x, p, t: -x,
        state=1,
        params=1,
        out=lambda x, p, t: jnp.concatenate([x, x, x]),
    )
    assert (problem.nstates, problem.nparams, problem.nout) == (1, 1, 3)
    assert solve(problem, Y0, jnp.array([1.0]), T0, T1, saveat=T_EVAL).shape == (5, 3)


@pytest.mark.parametrize("solver", ["bdf", "esdirk34", "tr_bdf2"])
def test_robertson(solver):
    problem = ROBERTSON.replace(solver=solver)
    ys = jax.jit(lambda y0, p: robertson(y0, p, jnp.array([0.0, 40.0]), problem))(
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
    # sensitivities), and jacrev with ad="forward" (their transpose). Both modes are per solve,
    # so they share one problem and one diffsol solver
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


def test_replace_builds_another_solver():
    other = LOGISTIC.replace(solver="tsit45")
    assert other is not LOGISTIC
    assert other.solver == "tsit45" and LOGISTIC.solver == "bdf"
    assert other._id != LOGISTIC._id
    p = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(
        solve(other, Y0, p, T0, T1, saveat=T_EVAL, **TOL), logistic(Y0, p), rtol=1e-7
    )
    with pytest.raises(DiffsolJaxError, match="unknown arguments"):
        LOGISTIC.replace(ode_solver="bdf")


def test_the_solver_is_freed_with_the_problem():
    problem = ODEProblem(lambda x, p, t: -x, state=1, params=1)
    finalizer = problem._finalizer
    assert finalizer.alive
    del problem
    assert not finalizer.alive


def test_evicted_checkpoint_is_recomputed():
    problem = LOGISTIC.replace(max_checkpoints=1)
    p = jnp.array([1.0, 1.0])

    def f(p):
        return solve(problem, Y0, p, T0, T1, saveat=T_EVAL, **TOL)

    ys, vjp = jax.vjp(f, p)
    (expected,) = vjp(jnp.ones_like(ys))
    assert problem.num_recomputes() == 0
    # a second forward solve frees the first checkpoint
    jax.vjp(f, jnp.array([2.0, 1.0]))
    (grad,) = vjp(jnp.ones_like(ys))
    assert problem.num_recomputes() == 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_grad_of_scan_with_more_solves_than_checkpoints():
    """`jax.grad` of a scan runs all the forward solves before the backward solves, so most of
    the checkpoints are freed; the backward solves run the forward solves again."""
    problem = LOGISTIC.replace(max_checkpoints=2)

    def f(p):
        def body(acc, r):
            return acc + jnp.sum(
                solve(problem, Y0, p * r, T0, T1, saveat=T_EVAL, **TOL) ** 2
            ), None

        return jax.lax.scan(body, 0.0, jnp.linspace(0.8, 1.2, 6))[0]

    p = jnp.array([1.0, 1.5])
    grad = jax.jit(jax.grad(f))(p)
    np.testing.assert_allclose(grad, finite_difference(f, p), rtol=1e-5)
    assert problem.num_recomputes() > 0


def _bkwd(problem, handle, params, ct, **over):
    """Call the backward primitive directly, as a wrong residual would reach it."""
    from diffsol_jax import _primitives

    kw = {"problem": problem, **PRIM_KW, **over}
    prim = _primitives.VjpBkwd(
        jax.typeof(handle),
        jax.typeof(params),
        jax.typeof(T0),
        jax.typeof(T_EVAL),
        jax.typeof(ct),
        **kw,
    )
    return prim(handle, params, T0, T_EVAL, ct)


def _fwd(problem, params, **over):
    """`params` here is the internal vector `[p0, p1, y0]` that the primitives take."""
    from diffsol_jax import _primitives

    kw = {"problem": problem, **PRIM_KW, **over}
    prim = _primitives._VjpFwdOp(
        jax.typeof(params), jax.typeof(T0), jax.typeof(T_EVAL), **kw
    )
    return prim(params, T0, T_EVAL)


def test_handle_with_other_params_is_not_used():
    p1, p2 = jnp.array([[1.0, 1.0, X0]]), jnp.array([[2.0, 1.5, X0]])
    ys, h1 = _fwd(LOGISTIC, p1)
    ct = jnp.ones_like(ys)
    expected = _bkwd(LOGISTIC, _fwd(LOGISTIC, p2)[1], p2, ct)
    before = LOGISTIC.num_recomputes()
    grad = _bkwd(LOGISTIC, h1, p2, ct)  # the handle of p1 with the params p2
    assert LOGISTIC.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_mixed_handles_are_not_used():
    p = jnp.array([[1.0, 1.0, X0], [2.0, 1.5, X0]])
    ys, h = _fwd(LOGISTIC, p)
    ct = jnp.ones_like(ys)
    expected = _bkwd(LOGISTIC, h, p, ct)
    # lane 1 has the id of a different forward solve
    _, other = _fwd(LOGISTIC, p[1:])
    before = LOGISTIC.num_recomputes()
    grad = _bkwd(LOGISTIC, h.at[1].set(other[0]), p, ct)
    assert LOGISTIC.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


P = jnp.array([1.0, 1.0])  # the logistic `r` and `k`


def test_max_steps_is_enforced():
    """A budget too small to reach `t1` fails the solve, rather than letting it run on."""
    with pytest.raises(Exception, match="maximum number of timesteps"):
        logistic(Y0, P, max_steps=1)


def test_max_steps_reaches_the_backward_solve():
    def loss(ms):
        return lambda p: logistic(Y0, p, max_steps=ms).sum()

    with pytest.raises(Exception, match="maximum number of timesteps"):
        jax.grad(loss(1))(P)
    # a generous budget is not a constraint: the same gradient as the default
    np.testing.assert_allclose(
        jax.grad(loss(100_000))(P), jax.grad(loss(10_000))(P), rtol=1e-12
    )


def test_max_steps_must_be_positive():
    for bad in (0, -1):
        with pytest.raises(DiffsolJaxError, match="max_steps must be at least 1"):
            logistic(Y0, P, max_steps=bad)
    with pytest.raises(TypeError):
        logistic(Y0, P, max_steps=1.5)


def test_handle_with_other_max_steps_is_not_used():
    """`max_steps` is part of the checkpoint key: a backward solve under a different budget
    runs the forward solve again rather than reusing the checkpoint."""
    p = jnp.array([[1.0, 1.0, X0]])
    ys, h = _fwd(LOGISTIC, p)
    ct = jnp.ones_like(ys)
    expected = _bkwd(LOGISTIC, h, p, ct)
    before = LOGISTIC.num_recomputes()
    grad = _bkwd(LOGISTIC, h, p, ct, max_steps=9_999)
    assert LOGISTIC.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_lanes_come_from_vmap_only():
    """An extra axis is a shape error, not a lane count: batching is `jax.vmap`."""
    p = jnp.array([1.0, 1.0])
    with pytest.raises(DiffsolJaxError, match="batch with jax.vmap"):
        solve(LOGISTIC, Y0, jnp.ones((4, 2)), T0, T1, saveat=T_EVAL)
    with pytest.raises(DiffsolJaxError, match="batch with jax.vmap"):
        solve(LOGISTIC, jnp.ones((3, 1)), p, T0, T1, saveat=T_EVAL)
    lanes = jax.vmap(lambda y0: logistic(y0, p))(jnp.ones((3, 1)) * X0)
    # one batched solve takes one step sequence for all its lanes, so it agrees with a loop of
    # single-lane solves to the solver accuracy, not to round-off
    np.testing.assert_allclose(lanes, jnp.stack([logistic(Y0, p)] * 3), rtol=1e-7)


def test_solve_argument_errors():
    p = jnp.array([1.0, 1.0])
    with pytest.raises(DiffsolJaxError, match="saveat must have one axis"):
        solve(LOGISTIC, Y0, p, T0, T1, saveat=jnp.ones((2, 2)))
    with pytest.raises(
        DiffsolJaxError, match=r"y0 has the shape \(2,\), expected \(1,\)"
    ):
        solve(LOGISTIC, jnp.ones(2), p, T0, T1, saveat=T_EVAL)
    with pytest.raises(DiffsolJaxError, match=r"params has the shape \(3,\)"):
        solve(LOGISTIC, Y0, jnp.ones(3), T0, T1, saveat=T_EVAL)
    with pytest.raises(DiffsolJaxError, match="ad must be"):
        solve(LOGISTIC, Y0, p, T0, T1, saveat=T_EVAL, ad="reverse")


def test_problem_argument_errors():
    rhs = lambda x, p, t: -x  # noqa: E731
    with pytest.raises(DiffsolJaxError, match="rhs must be a function"):
        ODEProblem(None, state=1)
    with pytest.raises(DiffsolJaxError, match="state must have at least one entry"):
        ODEProblem(rhs, state=None)
    with pytest.raises(DiffsolJaxError, match="state must have at least one entry"):
        ODEProblem(rhs, state=0)
    with pytest.raises(DiffsolJaxError, match="state must be an int or a pytree"):
        ODEProblem(rhs, state=True)
    with pytest.raises(DiffsolJaxError, match="every leaf of it must be one"):
        ODEProblem(rhs, state={"a": 1}, params=1)
    with pytest.raises(DiffsolJaxError, match="solver must be one of"):
        ODEProblem(rhs, state=1, params=1, solver="rk4")
    with pytest.raises(DiffsolJaxError, match="matrix_type must be one of"):
        ODEProblem(rhs, state=1, params=1, matrix_type="dense")
    with pytest.raises(DiffsolJaxError, match="max_checkpoints"):
        ODEProblem(rhs, state=1, params=1, max_checkpoints=0)
    with pytest.raises(DiffsolJaxError, match="params must not be negative"):
        ODEProblem(rhs, state=1, params=-1)
    with pytest.raises(DiffsolJaxError, match="every leaf of it must be one"):
        ODEProblem(rhs, state={"a": 1}, params=1)
    with pytest.raises(DiffsolJaxError, match="the result of rhs"):
        solve(
            ODEProblem(lambda y, p, t: y[:1] * p, state=2, params=1),
            jnp.ones(2),
            jnp.ones(1),
            T0,
            T1,
            saveat=T_EVAL,
        )
