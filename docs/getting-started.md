# Getting started

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python $\geq$ 3.12 and JAX $\geq$ 0.12.

## The PJRT plugin

The equation callbacks are compiled and executed through the
[PJRT C API](https://openxla.org/xla/pjrt), so diffsol-jax needs a CPU PJRT
plugin. jaxlib does not ship one as a loadable library. A prebuilt plugin for
linux/amd64 is inside [gopjrt](https://github.com/gomlx/gopjrt)'s
`gomlx_xlabuilder_linux_amd64.tar.gz` release asset (at
`lib/gomlx/pjrt/pjrt_c_api_cpu_plugin.so`; the macOS arm64 asset is
`gopjrt_cpu_darwin_arm64.tar.gz`), or you can build `pjrt_c_api_cpu_plugin.so`
from [openxla/xla](https://github.com/openxla/xla). Put it in
`~/.local/share/pjrt/`, where it is found automatically, or point the
environment variable at it:

```bash
export DIFFSOL_JAX_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

!!! tip "Building from source"

```text
Working against the development version needs a Rust toolchain (for
[diffsol-c](https://crates.io/crates/diffsol-c)), cmake, and a C++20 compiler. See the
README's *Building (development)* section for the exact commands.
```

## Enable 64-bit floats

The solver uses `f64`. JAX defaults to `f32`, so enable x64 once at startup:

```python
import jax
import jax.numpy as jnp
from diffsol_jax import Model

jax.config.update("jax_enable_x64", True)
```

## First forward solve

A model is defined by one plain `jax.numpy` function, the right-hand side
`rhs(x, p, t)`, with `x` of shape `[nstates]` and `p` of shape `[nparams]`. The
initial state is not part of the model: it is an argument of each solve.

```python
def lotka_volterra(x, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]]
    )


model = Model(
    rhs=lotka_volterra,
    nstates=2,
    nparams=4,
)
```

Solve from an initial state, over a vector of output times. `t_eval` lists the
points at which the solution is returned:

```python
y0 = jnp.array([1.0, 0.5])
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t_eval = jnp.linspace(0.0, 10.0, 200)
ys = model.solve(y0, params, t_eval)  # float64[200, 2]
```

`solve` works under `jax.jit`, and under `jax.vmap`: a vmapped axis of `y0` or of
`params` becomes more lanes in **one** batched solve, with the equations of all
the lanes evaluated together:

```python
ys = jax.jit(lambda p: model.solve(y0, p, t_eval))(params)
many = jax.vmap(model.solve, (None, 0, None))(y0, param_batch, t_eval)  # [B, 200, 2]
ics = jax.vmap(model.solve, (0, None, None))(y0_batch, params, t_eval)  # [B, 200, 2]
```

## Computing gradients

`jax.grad` uses the diffsol adjoint by default; `jax.jacfwd`/`jax.jvp` use
forward sensitivities. Both differentiate `y0` and `params`:

```python
def loss(p):
    return jnp.sum(model.solve(y0, p, t_eval) ** 2)


grad = jax.grad(loss)(params)
jac = jax.jacfwd(lambda p: model.solve(y0, p, t_eval))(params)
g_y0 = jax.grad(lambda y0: jnp.sum(model.solve(y0, params, t_eval) ** 2))(y0)
```

An initial state that depends on the parameters needs no special support — write
it as a function and let JAX compose:

```python
# y(0) = 1 and z(0) = p1, fitted together with p
jnp.sum(model.solve(jnp.array([1.0, p[1]]), p, t_eval) ** 2)
```

The maths, the `ad=` option, and checkpointing are described in
[Computing gradients](gradients.md).

## Systems: classes instead of flat vectors

For ML - a neural network inside the ODE. A `System` seeks to simplify the
setup: the **fields** are the differentiable inputs (a frozen dataclass,
registered as a JAX pytree), and the **method** `__call__(x, t)` is the
right-hand side:

```python
import equinox as eqx
from diffsol_jax import Model, System


class ExampleSystem(System):
    y0: jax.Array      # the initial state, trained with the weights
    net: eqx.Module    # any pytree of weights

    def __call__(self, x, t):
        return self.net(x)


net = SomeNeuralNetwork()
system = ExampleSystem(y0=jnp.ones(nstates), net=net)
model = Model(system, nstates=nstates, ode_solver="tsit45")
ys = model.solve(system.y0, system, t_eval)
```

Field leaves that are float arrays are trainable; every other leaf (ints, bools,
callables) is fixed when the model is built. `jax.grad` returns an
`ExampleSystem` of gradients:

```python
loss = lambda s: jnp.sum((model.solve(s.y0, s, t_eval) - data) ** 2)
grads = jax.grad(loss)(system) # an ExampleSystem, including grads.y0
updates, opt_state = opt.update(grads, opt_state, system)
system = optax.apply_updates(system, updates)
```

Holding `y0` as a field, as above, is what trains the initial state along with
the weights — it is a trainable leaf like any other, and passing it to `solve`
is what makes it the initial state. A `System` whose initial state is fixed
simply leaves it out and passes a constant.

A `System` can also override `out(x, t)` (an output function; `nout` is
inferred) and `mass(v, t)` (a mass matrix action for [DAEs](dae.md)).
`algebraic=` combines with a `System` as usual:
`Model(system, nstates=2, algebraic=1)`.

## Solver options

`Model` takes the solver configuration at construction:

| Argument            | Default            | Description                                                           |
| ------------------- | ------------------ | --------------------------------------------------------------------- |
| `ode_solver`        | `"bdf"`            | `"bdf"`, `"esdirk34"`, `"tr_bdf2"` (implicit), `"tsit45"` (explicit). |
| `rtol`, `atol`      | `1e-6`, `1e-6`     | Solver tolerances.                                                    |
| `ad`                | `"adjoint"`        | Reverse mode: `"adjoint"` or `"forward"`.  |
| `max_checkpoints`   | `8`                | Adjoint checkpoints kept.      |
| `out`, `nout`       | `None`, `0`        | Optional output function `out(x, p, t)`; `ys` then has `nout` columns.|
| `mass`, `algebraic` | `None`             | Mass matrix / DAE support - see [DAEs](dae.md).                       |
| `linear_solver`     | `"default"`        | `"lu"` or `"klu"` (with `matrix_type="faer_sparse"`).                 |
| `matrix_type`       | `"nalgebra_dense"` | `"faer_dense"` or `"faer_sparse"` for larger systems.                 |

## Where to go next

- [Computing gradients](gradients.md) - the adjoint and forward-sensitivity
  maths.
- [DAEs](dae.md) - algebraic constraints via singular mass matrices.
- [Architecture](architecture.md) - how a JAX function becomes a diffsol
  callback.
- [API reference](api/index.md) - full details on `Model`.
