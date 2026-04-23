import jax
import jax.numpy as jnp
from diffsol_jax import make_diffsol_solver

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
    solver, _ = make_diffsol_solver(
        lotka_volterra,
        y0=y0,
        p_example=params,
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
        n_times=100,
    )
    t_span = jnp.array([0.0, 10.0])

    def loss(p):
        ys, _ = solver(p, t_span)
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
    solver, _ = make_diffsol_solver(
        rhs_decay,
        y0=jnp.array([1.0]),
        p_example=jnp.array([k]),
        param_names=["k"],
        state_names=["x"],
        n_times=50,
    )
    t_span = jnp.array([0.0, T])

    def loss(p):
        ys, _ = solver(p, t_span)
        return ys[-1, 0] ** 2

    g = jax.grad(loss)(jnp.array([k]))[0]
    expected = -2.0 * T * jnp.exp(-2.0 * k * T)
    rel_err = abs(g - expected) / abs(expected)
    assert rel_err < 1e-3, f"closed-form err {rel_err}: got {g}, expected {expected}"


def test_lv_param_fitting():
    import optax

    true_p = jnp.array([1.5, 1.0, 0.75, 3.0])
    solver, _ = make_diffsol_solver(
        lotka_volterra,
        y0=jnp.array([1.0, 0.5]),
        p_example=true_p,
        ode_solver="tsit45",
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
        n_times=50,
    )
    t_span = jnp.array([0.0, 10.0])
    target_ys, _ = solver(true_p, t_span)

    def loss(p):
        ys, _ = solver(p, t_span)
        return jnp.mean((ys - target_ys) ** 2)

    p = true_p + 0.2 * jax.random.normal(jax.random.PRNGKey(0), (4,))
    opt = optax.adam(1e-2)
    state = opt.init(p)

    @jax.jit
    def step(p, state):
        l, g = jax.value_and_grad(loss)(p)
        updates, state = opt.update(g, state)
        return optax.apply_updates(p, updates), state, l

    for _ in range(500):
        p, state, l = step(p, state)

    assert jnp.max(jnp.abs(p - true_p)) < 0.05, (
        f"did not converge: p={p}, true={true_p}"
    )
