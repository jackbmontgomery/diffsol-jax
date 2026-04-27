import jax
import jax.numpy as jnp
import optax
from diffsol_jax import ODEProblem

jax.config.update("jax_enable_x64", True)


def rhs_decay(t, y, p):
    return (-p[0] * y[0],)


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def test_lv_grad_matches_fd():
    params = jnp.array([1.5, 1.0, 0.75, 3.0])
    y0 = jnp.array([1.0, 0.5])
    ode_problem = ODEProblem(lotka_volterra, y0, params)
    t_span = jnp.array([0.0, 10.0])

    def loss(p):
        _, ys = ode_problem.solve(p, t_span)
        return jnp.sum(ys**2)

    grad_ad = jax.grad(loss)(params)

    eps = 1e-5
    grad_fd = jnp.array(
        [
            (loss(params.at[i].add(eps)) - loss(params.at[i].add(-eps))) / (2 * eps)
            for i in range(4)
        ]
    )
    rel = jnp.linalg.norm(grad_ad - grad_fd) / jnp.linalg.norm(grad_fd)
    assert rel < 1e-3, f"rel err {rel}, ad={grad_ad}, fd={grad_fd}"


def test_decay_closed_form():
    T = 2.0
    k = 0.7
    ode_problem = ODEProblem(rhs_decay, jnp.array([1.0]), jnp.array([k]))
    t_span = jnp.array([0.0, T])

    def loss(p):
        _, ys = ode_problem.solve(p, t_span, ode_solver="tsit45")
        return ys[-1, 0] ** 2

    g = jax.grad(loss)(jnp.array([k]))[0]
    expected = -2.0 * T * jnp.exp(-2.0 * k * T)
    rel_err = abs(g - expected) / abs(expected)
    assert rel_err < 1e-3, f"closed-form err {rel_err}: got {g}, expected {expected}"


def test_lv_param_fitting():

    true_p = jnp.array([1.5, 1.0, 0.75, 3.0])
    y0 = jnp.array([1.0, 0.5])

    ode_problem = ODEProblem(lotka_volterra, y0, true_p)
    t_span = jnp.array([0.0, 10.0])
    _, target_ys = ode_problem.solve(true_p, t_span)

    def loss(p):
        _, ys = ode_problem.solve(p, t_span)
        return jnp.mean((ys - target_ys) ** 2)

    p = true_p + 0.2 * jax.random.normal(jax.random.PRNGKey(0), (4,))
    opt = optax.adam(1e-2)
    state = opt.init(p)

    @jax.jit
    def step(p, state):
        loss_val, grads = jax.value_and_grad(loss)(p)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(p, updates), state, loss_val

    for _ in range(500):
        p, state, loss_val = step(p, state)

    assert jnp.max(jnp.abs(p - true_p)) < 0.05, (
        f"did not converge: p={p}, true={true_p}"
    )
