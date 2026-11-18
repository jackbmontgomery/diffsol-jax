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
from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)
```

## First forward solve

A problem is the right-hand side `rhs(y, p, t)` — a plain `jax.numpy` function
of one lane — together with the layouts of the state and the parameters and the
solver options it is built with. The initial state is not part of it: that is an
argument of each solve.

```python
def lotka_volterra(y, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * y[0] - beta * y[0] * y[1], delta * y[0] * y[1] - gamma * y[1]]
    )


problem = ODEProblem(lotka_volterra, state=2, params=4)
```

`state=2` and `params=4` are the flat case: one `f64[2]` state and one `f64[4]`
parameter vector. Both can be pytrees instead — see
[Pytrees](#pytrees-the-state-and-the-parameters).

Constructing the problem builds the diffsol solver for it, and that solver — its
compiled callbacks, its adjoint checkpoints — lives as long as the problem, so
build it once, outside the loop that solves it. To solve the same equations with
a different solver, make another problem: `problem.replace(solver="tsit45")`.

Solve from an initial state, over a vector of output times. `t_eval` lists the
points at which the solution is returned:

```python
y0 = jnp.array([1.0, 0.5])
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t_eval = jnp.linspace(0.0, 10.0, 200)
ys = solve(problem, y0, t_eval, params)  # float64[200, 2]
```

`solve` works under `jax.jit`, and under `jax.vmap`: a vmapped axis of `y0` or
of `params` becomes more lanes in **one** batched solve, with the equations of
all the lanes evaluated together:

```python
ys = jax.jit(solve)(problem, y0, t_eval, params)
many = jax.vmap(solve, (None, None, None, 0))(problem, y0, t_eval, param_batch)
ics = jax.vmap(solve, (None, 0, None, None))(problem, y0_batch, t_eval, params)
```

`jax.vmap` is the *only* way to batch: the leaves of `y0` and `params` must have
the shapes the problem declares, so an extra axis is a shape error rather than a
silent lane count.

(A problem is a static pytree — it holds functions and layouts, never JAX arrays
— so it can be an argument of a jitted or vmapped function without being
traced.)

## Computing gradients

`jax.grad` uses the diffsol adjoint by default; `jax.jacfwd`/`jax.jvp` use
forward sensitivities. Both differentiate `y0` and the parameters:

```python
def loss(p):
    return jnp.sum(solve(problem, y0, t_eval, p) ** 2)


grad = jax.grad(loss)(params)
jac = jax.jacfwd(lambda p: solve(problem, y0, t_eval, p))(params)
g_y0 = jax.grad(lambda y0: jnp.sum(solve(problem, y0, t_eval, params) ** 2))(y0)
```

An initial state that depends on the parameters needs no special support — write
it as a function and let JAX compose:

```python
# y(0) = 1 and z(0) = p1, fitted together with p
jnp.sum(solve(problem, jnp.array([1.0, p[1]]), t_eval, p) ** 2)
```

The maths, the `ad=` option, and checkpointing are described in
[Computing gradients](gradients.md).

## Pytrees: the state and the parameters

`state=` and `params=` each take an int — one flat `f64[n]` array, as above — or
a **template** pytree whose float-array leaves give the shapes and dtypes. The
equations act on one lane of those pytrees.

For the state, every leaf must be a float array, since every leaf is integrated.
`y0`, what `rhs` returns, and `ys` all share the structure, so a system with
named blocks keeps its names:

```python
problem = ODEProblem(
    lambda y, p, t: {"pos": y["vel"], "vel": -p.k * y["pos"]},
    state={"pos": jnp.zeros(3), "vel": jnp.zeros(3)},
    params=Spring(k=jnp.array(1.0)),
)

ys = solve(problem, {"pos": pos0, "vel": vel0}, t_eval, params)
ys["pos"]  # [nt, 3]
ys["vel"]  # [nt, 3]
```

For the parameters, the float-array leaves are the parameters (differentiated,
in flatten order) and every other leaf (ints, bools, callables, the structure
itself) is captured by the equations as a constant, its value coming from the
template. That is what makes a neural ODE ordinary: the network is the
parameters.

```python
import equinox as eqx
import optax
from diffsol_jax import ODEProblem, solve

net = SomeNeuralNetwork()  # any pytree of weights
problem = ODEProblem(
    lambda y, net, t: net(y), state=nstates, params=net, solver="tsit45"
)
ys = solve(problem, y0, t_eval, net)
```

Only the shapes and dtypes of the parameter leaves are read from the template,
so `params=jax.eval_shape(make_net, key)` declares the layout without building
the weights at all. The problem never holds them; each solve passes the current
ones.

`jax.grad` returns a pytree of the same structure, each leaf's gradient in that
leaf's own dtype, so optax updates it directly:

```python
loss = lambda net: jnp.sum((solve(problem, y0, t_eval, net) - data) ** 2)
grads = jax.grad(loss)(net)
updates, opt_state = opt.update(grads, opt_state, net)
net = optax.apply_updates(net, updates)
```

To train the initial state together with the weights, put it in the same pytree
and pass it as both arguments — it is a parameter leaf like any other, and
passing it as `y0` is what makes it the initial state:

```python
class Fit(eqx.Module):
    y0: jax.Array
    net: eqx.Module


v = Fit(y0=jnp.ones(nstates), net=net)
problem = ODEProblem(lambda y, v, t: v.net(y), state=nstates, params=v)
ys = solve(problem, v.y0, t_eval, v)
grads = jax.grad(lambda v: loss(solve(problem, v.y0, t_eval, v)))(v)  # a Fit
```

Two notes on the structure:

- `y0` and `params` must match their templates: same structure, same leaf
  shapes. Batch with `jax.vmap` over the leaves, never by adding axes to them.
- the values of the **constant** parameter leaves come from the template, since
  they are compiled into the callbacks; the matching leaves of what a solve
  passes are ignored.

## The equations

Every equation takes one lane: `y` (or `v`) is in the state structure, `p` is in
the parameter structure, and `t` is a scalar. Any traceable JAX code is allowed.

| Argument    | Signature   | Description                                                  |
| ----------- | ----------- | ------------------------------------------------------------ |
| `rhs`       | `(y, p, t)` | The right-hand side, returning the state structure.          |
| `out`       | `(y, p, t)` | Optional: an output returning any pytree, which `ys` then comes back in instead of the state. Its structure is inferred. |
| `mass`      | `(v, p, t)` | Optional: the mass matrix action `M(t) v`, with `v` in the state structure, linear in `v` and not using `p` — see [DAEs](dae.md). |
| `algebraic` |             | Optional: a `mass` shortcut for a semi-explicit DAE, an int, a boolean mask, or a pytree of bools matching the state. |

## Solver options

The four options the diffsol solver is built with belong to the problem:

| Argument          | Default            | Description                                                           |
| ----------------- | ------------------ | --------------------------------------------------------------------- |
| `solver`          | `"bdf"`            | `"bdf"`, `"esdirk34"`, `"tr_bdf2"` (implicit), `"tsit45"` (explicit). |
| `max_checkpoints` | `8`                | Adjoint checkpoints kept.                                             |
| `linear_solver`   | `"default"`        | `"lu"` or `"klu"` (with `matrix_type="faer_sparse"`).                 |
| `matrix_type`     | `"nalgebra_dense"` | `"faer_dense"` or `"faer_sparse"` for larger systems.                 |

The three that are per solve belong to `solve`:

| Argument       | Default        | Description                                     |
| -------------- | -------------- | ----------------------------------------------- |
| `rtol`, `atol` | `1e-6`, `1e-6` | Solver tolerances.                              |
| `ad`           | `"adjoint"`    | Reverse mode: `"adjoint"` or `"forward"`.       |

So comparing the two reverse modes, or tightening a tolerance, needs no second
problem; changing the solver does (`problem.replace(solver=...)`), because that
is a second diffsol solver.

## Where to go next

- [Computing gradients](gradients.md) - the adjoint and forward-sensitivity
  maths.
- [DAEs](dae.md) - algebraic constraints via singular mass matrices.
- [Architecture](architecture.md) - how a JAX function becomes a diffsol
  callback.
- [API reference](api/index.md) - full details on `ODEProblem` and `solve`.
