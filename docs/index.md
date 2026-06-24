# diffsol-jax

diffsol-jax is a [JAX](https://docs.jax.dev/) wrapper around
[diffsol](https://github.com/martinjrobins/diffsol), a fast Rust ODE solver library.

You write an ODE right-hand side as an ordinary `jax.numpy` function. diffsol-jax traces it, lowers
it to [DiffSL](https://martinjrobins.github.io/diffsl/), and JIT-compiles it with
[Cranelift](https://cranelift.dev/) into native code that diffsol runs. The result is a `solve`
callable that plugs straight into the JAX ecosystem:

- it runs inside `jax.jit`;
- it differentiates under `jax.grad`, `jax.jvp`, `jax.jacfwd`, and `jax.vjp`;
- it batches under `jax.vmap`.

The solver loop itself lives entirely in compiled Rust, so there is no Python or XLA tracing
overhead per step. On stiff problems this is often **orders of magnitude** faster than pure-JAX
solvers.

`diffsol-jax` is built for **fast forward solves and fast parameter gradient descent for small ODE
systems running on CPU.** If you are fitting the parameters of a small, stiff or non-stiff system,
this is the library for you.

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python $\geq$ 3.11 and JAX.

## Quick example

```python
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem

jax.config.update("jax_enable_x64", True) # Using f64 is recommended


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return jnp.array([alpha * x - beta * x * yy, delta * x * yy - gamma * yy])


params = jnp.array([1.5, 1.0, 0.75, 3.0])
y0 = jnp.array([1.0, 0.5])

problem = ODEProblem(lotka_volterra, y0=y0, params=params)

t_eval = jnp.linspace(0.0, 10.0, 200)
ts, ys = problem.solve(params, t_eval)


def loss(p):
    _, ys = problem.solve(p, t_eval)
    return jnp.sum(ys**2)


grad = jax.grad(loss)(params)  # gradient w.r.t. parameters
```

## Sharp bits :knife:

- **Use `f64`.** diffsol is built for `f64` like most scientific solvers; JAX defaults to `f32`.
  Enable x64 (`jax.config.update("jax_enable_x64", True)`). This is also significantly faster than
  the native-JAX solvers.
- **No higher-order derivatives.** `grad(grad(solve(...)))` does not work in general.
- **`y0` is fixed.** The initial state is baked into the compiled problem at construction time and
  cannot change between `solve` calls; gradients are not taken w.r.t. `y0`.
- **Neural ODEs** are not supported since this requires operations that are not currently supported
  by the lowering.
- **Only CPU is supported**.

## Next steps

- [Getting started](getting-started.md): install, first forward solve, first gradient.
- [Computing gradients](gradients.md): the forward-sensitivity maths behind `jax.grad`.
- [From Python to DiffSL](lowering.md): how your `jax.numpy` RHS becomes a compiled solver.
- [Benchmarks](benchmarks.md): forward and gradient timing vs diffrax.
- [API reference](api/index.md): `ODEProblem`, solver options, type aliases.

## See also

Other libraries in the JAX scientific ecosystem:

- [diffrax](https://github.com/patrick-kidger/diffrax): differential equation solvers in pure JAX.
- [equinox](https://github.com/patrick-kidger/equinox): neural networks and PyTree tooling.
- [optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, least
  squares.
