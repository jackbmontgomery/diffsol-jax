# diffsol-jax

JAX bindings for the Rust-based ordinary differential equations and
semi-explicit differential algebraic equations solving library
[diffsol](https://github.com/martinjrobins/diffsol)

## Key Features

- FFI calls are wrapped as primitive operations supporting `jacfwd` and
  `jacrev`;
- Double and single precision;
- Forward and adjoint gradient computations;
- Semi-explicit DAE support;
- Vectorized solves are solved as lanes on a single solve;
- Using PyTrees as the state;
- Support for neural differential equations;

## Sharp Bits

- **CPU only, for now.**
- **No higher-order derivatives.** `grad(grad(solve(...)))` does not work.
- **Small non-stiff performance.** Each solver callback costs ~2.5 µs of fixed
  PJRT overhead; an explicit method on a tiny system is often faster in
  `diffrax`.

## Installation

```bash
pip install "diffsol-jax[cpu]"
```

## Documentation

<https://jackbmontgomery.github.io/diffsol-jax/>

## Quick Example

```python
import jax, jax.numpy as jnp
from diffsol_jax import ODEProblem, solve


def rhs(y, p, t):
    return p[0] * y * (1.0 - y)


N = 50
y0, p = jnp.array([0.1]), jnp.array([1.1])
t0, t1 = 0.0, 5.0
ts = jnp.linspace(t0, t1, N)

problem = ODEProblem(rhs, state=1, params=1)
ys = solve(problem, y0, p, t0, t1, saveat=ts)  # [N, nstates]
```

## Next Steps

[Getting Started](getting-started.md)

## See Also: Other Libraries in the JAX Ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and
everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for
shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam,...)
optimisers.  
[paramax](https://github.com/danielward27/paramax): parameterizations and
constraints for PyTrees.  

**Scientific computing**  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding,
minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian
sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX
conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX
honourable mention!)  
[diffrax](https://github.com/patrick-kidger/diffrax): numerical differential
equation solvers.

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other
JAX projects.  

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
- the state and the parameters are **pytrees**, so a system can keep its named
  blocks and a neural ODE's parameters can be
  [the network itself](getting-started.md#pytrees-the-state-and-the-parameters).

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python $\geq$ 3.12 and JAX $\geq$ 0.12, plus a CPU PJRT plugin, which
the `cpu` extra installs: `pip install "diffsol-jax[cpu]"` (see
[Getting started](getting-started.md)).

## Quick example

```python
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem, solve


def lotka_volterra(x, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]]
    )


problem = ODEProblem(lotka_volterra, state=2, params=4)

y0 = jnp.array([1.0, 0.5])
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t0, t1 = 0.0, 10.0
saveat = jnp.linspace(t0, t1, 200)
ys = solve(problem, y0, params, t0, t1, saveat=saveat)  # [200, 2]


def loss(p):
    return jnp.sum(solve(problem, y0, p, t0, t1, saveat=saveat) ** 2)


grad = jax.grad(loss)(params)  # adjoint gradient w.r.t. the parameters
many = jax.vmap(solve, (None, None, 0, None, None))(
    problem, y0, jnp.stack([params] * 64), t0, t1
)  # one solve, 64 lanes
```

A problem holds the equations, the layouts and the solver; a solve takes the
state, the times and the parameter values. Both layouts are pytrees: `state=2`
and `params=4` above are shorthand for one flat `[2]` and one flat `[4]` array,
and a template declares anything else. For ML - a neural network inside the ODE,
trained by gradient descent - the parameters are the network:

```python
problem = ODEProblem(lambda y, net, t: net(y), state=4, params=net, solver="tsit45")

ys = solve(problem, y0, net, t0, t1, saveat=saveat)  # the current weights
grads = jax.grad(lambda net: loss(solve(problem, y0, net, t0, t1, saveat=saveat)))(net)
```

`grads` is a network of gradients, which optax applies directly. See
[Pytrees](getting-started.md#pytrees-the-state-and-the-parameters) for the
rules, and the `examples/neural-ode-weather-prediction` example for a full
training loop.

## Next steps

- [Getting started](getting-started.md): install, first forward solve, first
  gradient.
- [Computing gradients](gradients.md): the adjoint and forward-sensitivity maths
  behind `jax.grad` and `jax.jacfwd`.
- [DAEs](dae.md): singular mass matrices and algebraic constraints.
- [Architecture](architecture.md): how the pieces fit together, with benchmarks.
- [API reference](api/index.md): `ODEProblem`, `solve` and their options.
