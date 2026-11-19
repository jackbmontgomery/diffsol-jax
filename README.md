<p align="center">
  <img src="assets/diffsol-jax.png" alt="Logo" style="width: 75%;">
</p>

JAX bindings for [diffsol](https://github.com/martinjrobins/diffsol). Write the
ODE equations as ordinary JAX functions solve them with diffsol while staying
compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd` and
`jax.jacrev`.

## Key features

## Example usage

```python
import jax, jax.numpy as jnp
from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)

problem = ODEProblem(lambda y, p, t: p[0] * y * (1.0 - y), state=1, params=1)
y0, p = jnp.array([0.1]), jnp.array([1.0])
t0, t1 = 0.0, 5.0
ys = solve(problem, y0, p, t0, t1)  # [1, nstates], the solution at t1
ts = jnp.linspace(t0, t1, 11)
ys = solve(problem, y0, p, t0, t1, saveat=ts)  # [11, nstates]
ys = jax.vmap(solve, (None, None, 0, None, None))(
    problem, y0, jnp.ones((64, 1)), t0, t1
)
g_y0, g_p = jax.grad(lambda y, p: solve(problem, y, p, t0, t1).sum(), (0, 1))(y0, p)
```

An `ODEProblem` is the equations plus the layouts and solver options they are
built into: constructing one builds the diffsol solver, with its compiled
callbacks and its adjoint checkpoints, and that solver lives as long as the
problem. A solve then supplies the values —
`solve(problem, y0, params, t0, t1, saveat=..., rtol=..., atol=..., ad=...)`.
`y0` is the state at `t0` and the solve ends at `t1`; by default the solution
comes back at `t1` alone, and `saveat` asks for it at other times in `[t0, t1]`.
To change the solver, make another problem:
`problem.replace(solver="tsit45")`.

`solve` is differentiable in both `y0` and `params`. Reverse mode
(`ad="adjoint"`, the default) uses the diffsol adjoint with checkpointing;
forward mode (`ad="forward"`) uses diffsol forward sensitivities. Batching is
`jax.vmap`: a vmapped axis of `y0` or of `params` becomes more lanes in a single
batched solve, not a loop of solves, so an ensemble over initial conditions
costs one solve.

An initial state that is itself a function of the parameters is ordinary JAX
composition: `solve(problem, f(p), p, t0, t1)` differentiates through `f`
without any special case.

## Pytrees

The state and the parameters are both pytrees, declared once by `state=` and
`params=`. Each is either an int `n`, meaning one flat `f64[n]` array, or a
template: its float-array leaves give the shapes and dtypes. The equations, `y0`
and `ys` all share the state structure, so a system with named blocks can stay
named:

```python
problem = ODEProblem(
    lambda y, p, t: {"pos": y["vel"], "vel": -p.k * y["pos"]},
    state={"pos": jnp.zeros(3), "vel": jnp.zeros(3)},
    params=Spring(k=jnp.array(1.0)),
)

ys = solve(problem, {"pos": pos0, "vel": vel0}, params, t0, t1, saveat=ts)
ys["pos"]  # [nt, 3]
```

In a `params` template, the float-array leaves are the parameters and every
other leaf is captured by the equations as a constant. `jax.grad` returns a
pytree of the same structure, so a network inside the right-hand side is
ordinary — and optax updates the gradients directly:

```python
import equinox as eqx

net = eqx.nn.MLP(4, 4, 64, 2, key=jax.random.key(0))  # any pytree of weights
problem = ODEProblem(lambda y, net, t: net(y), state=4, params=net, solver="tsit45")


def loss(net):
    ys = solve(problem, y0, net, t0, t1, saveat=ts)  # the current weights
    return jnp.sum((ys - data) ** 2)


grads = jax.grad(loss)(net)  # an MLP of gradients
net = optax.apply_updates(net, optimizer.update(grads, opt_state, net)[0])
```

Only the shapes and dtypes of the parameter leaves are read, so
`params=jax.eval_shape(make_net, key)` is a complete declaration — the problem
never holds the weights, each solve passes them. To train the initial state with
them, make it a leaf of the same pytree and pass it as both:
`solve(problem, v.y0, v, t0, t1)`. For full training loops, see
[examples/neural-ode-weather-prediction](examples/neural-ode-weather-prediction)
and
[examples/fitting-a-predator-prey-model](examples/fitting-a-predator-prey-model).

## DAEs

A singular mass matrix gives a semi-explicit index-1 DAE, solved with the
implicit solvers (`bdf`, the default, `esdirk34` or `tr_bdf2`) and
differentiable like any other problem. Write the algebraic equations as rows of
`rhs` and mark them with `algebraic=` — an int (the last entries of the flat
state), a boolean mask, or a pytree of bools matching the state:

```python
# y' = -p0 * y   and   0 = z - p1 * y
problem = ODEProblem(
    rhs=lambda y, p, t: jnp.array([-p[0] * y[0], y[1] - p[1] * y[0]]),
    state=2,
    params=2,
    algebraic=1,
)
p = jnp.array([0.7, 2.0])
ys = solve(problem, jnp.array([1.0, p[1]]), p, t0, t1)  # a consistent y0
```

The initial state is projected onto the algebraic constraints automatically
(diffsol solves the algebraic variables at `t0`), and the gradients account for
it — so the algebraic entries of the `y0` you pass are only a starting guess.
For a general (time-dependent) mass matrix, give `mass=lambda v, p, t: ...`, the
action `M(t) v`, which must be linear in `v` and must not use `p` (the diffsol
sensitivities have no `dM/dp` term, so a parameter-dependent mass would give a
wrong gradient and is rejected — fold such parameters into `rhs` instead).
Higher-index DAEs must be reduced to index 1 first.

## Building (development)

Requirements: Python ≥ 3.12, a Rust toolchain, cmake and a C++20 compiler, and a
CPU PJRT plugin.

1. Build diffsol-c from a [diffsol](https://github.com/martinjrobins/diffsol)
   checkout:

   ```sh
   cargo build -p diffsol-c --release
   ```

2. Set up the Python environment and build `libdiffsol_jax`:

   ```sh
   uv sync --group dev
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
     -DDIFFSOL_C_DIR=/path/to/diffsol/target/release \
     -DXLA_FFI_INCLUDE_DIR=$(.venv/bin/python -c "import jax.ffi; print(jax.ffi.include_dir())")
   cmake --build build
   ```

3. Get a CPU PJRT plugin. jaxlib does not ship one; a prebuilt plugin for
   linux/amd64 is inside [gopjrt](https://github.com/gomlx/gopjrt)'s
   `gomlx_xlabuilder_linux_amd64.tar.gz` release asset
   (`lib/gomlx/pjrt/pjrt_c_api_cpu_plugin.so`), or build
   `pjrt_c_api_cpu_plugin.so` from
   [openxla/xla](https://github.com/openxla/xla). Put it in
   `~/.local/share/pjrt/` (found automatically), or point
   `DIFFSOL_JAX_PJRT_PLUGIN` at it.

4. Make the package importable from the venv (once per venv):

   ```sh
   echo "$PWD/python" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/diffsol_jax_dev.pth"
   ```

   After this, `uv run python` and `.venv/bin/python` import `diffsol_jax` from
   `python/`, with the native library from `build/` — no `sys.path` changes
   needed in scripts.

5. Run the tests:

   ```sh
   .venv/bin/python -m pytest tests
   ```

## Status

This branch is a rewrite onto the PJRT architecture (vendored from the diffsol
[`diffsol-jax2` example](https://github.com/martinjrobins/diffsol/tree/diffsol-jax2/examples/diffsol-jax),
see [issue #303](https://github.com/martinjrobins/diffsol/issues/303)). CPU and
`f64` only for now; GPU arrives with diffsol's GPU backend by loading the CUDA
PJRT plugin instead.
