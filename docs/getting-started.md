# Getting Started

This page should take you through basic usage of `diffsol-jax`. See "Examples"
on the left for in-depth usage of the wrapper.

## Installation

```bash
pip install diffsol-jax[cpu]
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax[cpu]
```

Requires Python $\geq$ 3.12 and JAX $\geq$ 0.12.

> [!NOTE]
> The "cpu" extra is not strictly needed. It is included to download a
> [PJRT C API](https://openxla.org/xla/pjrt) as well as the wrapper. To use
> another plugin instead, one you built yourself from
> [openxla/xla](https://github.com/openxla/xla) for example, or a prebuilt
> `pjrt_c_api_cpu_plugin.so` from [gopjrt](https://github.com/gomlx/gopjrt)'s,
> point the environment variable at it:

```bash
export DIFFSOL_JAX_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so

```

## Problem Definition

`diffsol-jax` calls `diffsol` over the
[JAX foreign function interface (FFI)](https://docs.jax.dev/en/latest/401/ffi.html).

## First forward solve

```python
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem, solve
```

A problem is the right-hand side `rhs(y, p, t)` — a plain `jax.numpy` function
of one lane — together with the layouts of the state and the parameters and the
solver options it is built with. The initial state is not part of it: that is an
argument of each solve. By default a problem runs in `float32`, matching JAX's
own default -- nothing else to configure; see [Precision](#precision) for
`float64`.

```python
def lotka_volterra(y, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * y[0] - beta * y[0] * y[1], delta * y[0] * y[1] - gamma * y[1]]
    )


problem = ODEProblem(lotka_volterra, state=2, params=4)
```

`state=2` and `params=4` are the flat case: one `[2]` state and one `[4]`
parameter vector. Both can be pytrees instead — see
[Pytrees](#pytrees-the-state-and-the-parameters).

Constructing the problem builds the diffsol solver for it, and that solver — its
compiled callbacks, its adjoint checkpoints — lives as long as the problem, so
build it once, outside the loop that solves it. To solve the same equations with
a different solver, make another problem: `problem.replace(solver="tsit45")`.

Solve from the state `y0` at `t0` to `t1`. `y0` belongs to `t0`, and by default
the solution comes back at `t1` alone:

```python
y0 = jnp.array([1.0, 0.5])
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t0, t1 = 0.0, 10.0
ys = solve(problem, y0, params, t0, t1)  # float32[1, 2]
```

`saveat` asks for the solution at other times, each of them in `[t0, t1]`:

```python
saveat = jnp.linspace(t0, t1, 200)
ys = solve(problem, y0, params, t0, t1, saveat=saveat)  # float32[200, 2]
```

The window `saveat` asks for need not reach back to `t0`, so a solve can warm up
before the part you want to look at:

```python
ys = solve(problem, y0, params, 0.0, 10.0, saveat=jnp.linspace(5.0, 10.0, 100))
```

Here `y0` is the state at `t = 0`, the solver integrates from there, and only
the window `[5, 10]` comes back. The solve ends at the last time it returns, so
a `saveat` that stops short of `t1` does less work than one that reaches it.

`solve` works under `jax.jit`, and under `jax.vmap`: a vmapped axis of `y0` or
of `params` becomes more lanes in **one** batched solve, with the equations of
all the lanes evaluated together:

```python
ys = jax.jit(solve)(problem, y0, params, t0, t1)
many = jax.vmap(solve, (None, None, 0, None, None))(problem, y0, param_batch, t0, t1)
ics = jax.vmap(solve, (None, 0, None, None, None))(problem, y0_batch, params, t0, t1)
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
    return jnp.sum(solve(problem, y0, p, t0, t1) ** 2)


grad = jax.grad(loss)(params)
jac = jax.jacfwd(lambda p: solve(problem, y0, p, t0, t1))(params)
g_y0 = jax.grad(lambda y0: jnp.sum(solve(problem, y0, params, t0, t1) ** 2))(y0)
```

An initial state that depends on the parameters needs no special support — write
it as a function and let JAX compose:

```python
# y(0) = 1 and z(0) = p1, fitted together with p
jnp.sum(solve(problem, jnp.array([1.0, p[1]]), p, t0, t1) ** 2)
```

The maths, the `ad=` option, and checkpointing are described in
[Computing gradients](gradients.md).

## Pytrees: the state and the parameters

`state=` and `params=` each take an int — one flat `[n]` array, as above — or a
**template** pytree whose float-array leaves give the shapes and dtypes. The
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

ys = solve(problem, {"pos": pos0, "vel": vel0}, params, t0, t1)
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
ys = solve(problem, y0, net, t0, t1)
```

Only the shapes and dtypes of the parameter leaves are read from the template,
so `params=jax.eval_shape(make_net, key)` declares the layout without building
the weights at all. The problem never holds them; each solve passes the current
ones.

`jax.grad` returns a pytree of the same structure, each leaf's gradient in that
leaf's own dtype, so optax updates it directly:

```python
loss = lambda net: jnp.sum((solve(problem, y0, net, t0, t1) - data) ** 2)
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
ys = solve(problem, v.y0, v, t0, t1)
grads = jax.grad(lambda v: loss(solve(problem, v.y0, v, t0, t1)))(v)  # a Fit
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
| `dtype`           | `jnp.float32`      | `jnp.float64` runs the solver in double precision -- see [Precision](#precision). |
| `max_checkpoints` | `8`                | Adjoint checkpoints kept.                                             |
| `linear_solver`   | `"default"`        | `"lu"` or `"klu"` (with `matrix_type="faer_sparse"`).                 |
| `matrix_type`     | `"nalgebra_dense"` | `"faer_dense"` or `"faer_sparse"` for larger systems.                 |

The ones that are per solve belong to `solve`:

| Argument       | Default        | Description                                     |
| -------------- | -------------- | ----------------------------------------------- |
| `rtol`, `atol` | by dtype       | Solver tolerances: `1e-6` for an `f64` problem, `1e-4` for `f32`. |
| `max_steps`    | `10000`        | The most time steps one solve may take.         |
| `ad`           | `"adjoint"`    | Reverse mode: `"adjoint"` or `"forward"`.       |

So comparing the two reverse modes, or tightening a tolerance, needs no second
problem; changing the solver does (`problem.replace(solver=...)`), because that
is a second diffsol solver.

## Precision

A problem is `float32` by default, matching JAX's own default config, and runs
the solver, its callbacks, and `y0`, `params`, `ys`, `t0`, `t1`, `saveat` and
the gradients all in `f32` -- no JAX config needed at all:

```python
problem = ODEProblem(rhs, state=1, params=2)  # dtype=jnp.float32, implicitly
ys = solve(problem, y0, p, t0, t1, saveat=ts)  # f32 in, f32 out
```

For `float64`, give it explicitly *and* enable JAX's x64 config -- both are
required, since `float64` is an opt-in on both sides:

```python
jax.config.update("jax_enable_x64", True)

problem = ODEProblem(rhs, state=1, params=2, dtype=jnp.float64)
ys = solve(problem, y0, p, t0, t1, saveat=ts)  # f64 in, f64 out
```

Building a `float64` problem without `jax_enable_x64` raises
`DiffsolJaxError: dtype=float64 needs jax_enable_x64` -- the two must agree,
since there is no other way to get a value a JAX array can actually hold at that
precision.

A value you pass in is cast to the problem's dtype regardless of its own, so an
`f64` array is accepted by an `f32` problem and narrowed (and vice versa).

`f32` has an eps of 1.2e-7, so `rtol` and `atol` default to `1e-4` for an `f32`
problem rather than `1e-6`. Around `1e-5` is as tight as `f32` goes: below that
the forward solve may still manage, but the adjoint's Newton iteration diverges.
If a gradient fails with "Newton iteration diverged", loosen the tolerance or
use `float64` -- this gets more likely for a stiff problem at a large batch
size, where a handful of lanes can be stiffer than the rest.

In exchange, `f32` is substantially faster: roughly 2-5x on both the forward
solve and the gradient, growing with the batch size and the state size (fewer
steps at the looser default tolerance, and half the memory traffic per state
element). See [Benchmarks](benchmarks.md).

## Where to go next

- [Computing gradients](gradients.md) - the adjoint and forward-sensitivity
  maths.
- [DAEs](dae.md) - algebraic constraints via singular mass matrices.
- [Architecture](architecture.md) - how a JAX function becomes a diffsol
  callback.
- [API reference](api/index.md) - full details on `ODEProblem` and `solve`.
