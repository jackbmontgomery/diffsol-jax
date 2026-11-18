# diffsol-jax

diffsol-jax is a [JAX](https://docs.jax.dev/) wrapper around
[diffsol](https://github.com/martinjrobins/diffsol), a fast Rust ODE/DAE solver
library.

You write the ODE equations as ordinary `jax.numpy` functions. diffsol-jax
derives every callback the solver needs with JAX's own transformations, lowers
each to StableHLO, and executes them natively through the PJRT C API. The solves
run in diffsol and are exposed to JAX as XLA FFI custom calls. The result is a
`solve` that plugs straight into the JAX ecosystem:

- it runs inside `jax.jit`;
- it differentiates under `jax.grad`, `jax.jacrev`, `jax.jvp`, and `jax.jacfwd`;
- it batches under `jax.vmap` - a vmapped axis becomes more lanes in a single
  batched solve;
- it solves **DAEs**;
- it takes the parameters as a **pytree**: a network's weights, or anything
  else [optax can update](getting-started.md#parameters-as-a-pytree), for neural
  ODEs.

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python $\geq$ 3.12 and JAX $\geq$ 0.12, plus a CPU PJRT plugin (see
[Getting started](getting-started.md)).

## Quick example

```python
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)  # the solver uses f64


def lotka_volterra(x, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]]
    )


problem = ODEProblem(lotka_volterra)

y0 = jnp.array([1.0, 0.5])
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t_eval = jnp.linspace(0.0, 10.0, 200)
ys = solve(problem, y0, t_eval, params)  # [200, 2]


def loss(p):
    return jnp.sum(solve(problem, y0, t_eval, p) ** 2)


grad = jax.grad(loss)(params)  # adjoint gradient w.r.t. the parameters
many = jax.vmap(solve, (None, None, None, 0))(
    problem, y0, t_eval, jnp.stack([params] * 64)
)  # one solve, 64 lanes
```

A problem holds the equations; a solve takes the state, the times, the
parameters and the solver options. For ML — a neural network inside the ODE,
trained by gradient descent — the parameters are the network:

```python
problem = ODEProblem(lambda x, net, t: net(x))

ys = solve(problem, y0, t_eval, net, solver="tsit45")
grads = jax.grad(lambda net: loss(solve(problem, y0, t_eval, net)))(net)
```

`grads` is a network of gradients, which optax applies directly. See
[Parameters as a pytree](getting-started.md#parameters-as-a-pytree) for the
rules, and the `examples/neural-ode-weather-prediction` example for a full
training loop.

## Sharp bits :knife

- **`f64` is required.** diffsol is built for `f64` like most scientific
  solvers; enable x64 (`jax.config.update("jax_enable_x64", True)`) before the
  first solve.
- **A CPU PJRT plugin is required.** jaxlib does not ship one; put it in
  `~/.local/share/pjrt/` (found automatically) or set
  `DIFFSOL_JAX_PJRT_PLUGIN` to its path (see
  [Getting started](getting-started.md)).
- **No higher-order derivatives.** `grad(grad(solve(...)))` does not work.
- **No derivative with respect to `t_eval`.** Output times are constants.
- **CPU only, for now.** The PJRT C API is backend-agnostic, so GPU arrives with
  diffsol's GPU backend by loading the CUDA PJRT plugin instead.
- **Small non-stiff systems are not fast.** Each solver callback costs ~2.5 µs
  of fixed PJRT overhead; an explicit method on a tiny system is often faster in
  pure JAX ([diffrax](https://github.com/patrick-kidger/diffrax)). Stiff or
  batched problems are where diffsol-jax is better suited - see
  [Benchmarks](benchmarks.md).

## Next steps

- [Getting started](getting-started.md): install, first forward solve, first
  gradient.
- [Computing gradients](gradients.md): the adjoint and forward-sensitivity maths
  behind `jax.grad` and `jax.jacfwd`.
- [DAEs](dae.md): singular mass matrices and algebraic constraints.
- [Architecture](architecture.md): how the pieces fit together, with benchmarks.
- [API reference](api/index.md): `ODEProblem`, `solve` and their options.

## See also

Other libraries in the JAX scientific ecosystem:

- [diffrax](https://github.com/patrick-kidger/diffrax): differential equation
  solvers in pure JAX.
- [equinox](https://github.com/patrick-kidger/equinox): neural networks and
  PyTree tooling.
- [optimistix](https://github.com/patrick-kidger/optimistix): root finding,
  minimisation, least squares.
