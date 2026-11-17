<p align="center">
  <img src="assets/diffsol-jax.png" alt="Logo" style="width: 75%;">
</p>

Use [diffsol](https://github.com/martinjrobins/diffsol) solves as JAX operations. Write the ODE
equations as ordinary JAX functions and use the solves in `jax.jit`, `jax.vmap`, `jax.grad`,
`jax.jacfwd` and `jax.jacrev`.

The equations stay in JAX: every callback diffsol needs (`rhs`, its Jacobian action, the adjoint,
the sensitivities, ...) is derived with JAX's own transformations (`jax.jvp` / `jax.vjp`), lowered
to StableHLO, and executed natively through the PJRT C API. The solves run in diffsol
([diffsol-c](https://crates.io/crates/diffsol-c)) and are exposed to JAX as XLA FFI custom calls.
There is no translation of your Python to another language, so anything JAX can trace works as an
equation. See [docs/architecture.md](docs/architecture.md) for the full design and benchmarks.

## Example usage

```python
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from diffsol_jax import Model

model = Model(
    rhs=lambda x, p, t: p[0] * x * (1.0 - x),  # one lane: x [nstates], p [nparams]
    nstates=1,
    nparams=1,
)
y0, p = jnp.array([0.1]), jnp.array([1.0])
t = jnp.linspace(0.0, 5.0, 11)
ys = model.solve(y0, p, t)  # [nt, nstates]
ys = jax.vmap(model.solve, (None, 0, None))(y0, jnp.ones((64, 1)), t)  # 64 lanes, one solve
g_y0, g_p = jax.grad(lambda y0, p: model.solve(y0, p, t).sum(), (0, 1))(y0, p)
```

`solve(y0, params, t_eval)` is differentiable in both `y0` and `params`. Reverse mode
(`ad="adjoint"`, the default) uses the diffsol adjoint with checkpointing; forward mode
(`ad="forward"`) uses diffsol forward sensitivities. A `jax.vmap` over `y0` or `params` becomes
more lanes in a single batched solve, not a loop of solves — so an ensemble over initial
conditions costs one solve.

An initial state that is itself a function of the parameters is ordinary JAX composition:
`model.solve(f(p), p, t)` differentiates through `f` without any special case.

## Neural ODEs

For ML, define the problem as one `System`: the fields are the differentiable inputs (a frozen
dataclass pytree — an initial state, a network, rate constants) and the methods are the
equations. `jax.grad` returns a pytree of the same structure, so optax updates it directly:

```python
from diffsol_jax import Model, System

class WeatherSystem(System):
    y0: jax.Array    # the initial state, trained with everything else
    net: eqx.Module  # any pytree of weights

    def __call__(self, x, t):
        return self.net(x)

system = WeatherSystem(y0=y0, net=net)
model = Model(system, nstates=4, ode_solver="tsit45")

loss = lambda s: jnp.sum((model.solve(s.y0, s, t_eval) - data) ** 2)
grads = jax.grad(loss)(system)              # a WeatherSystem of gradients
system = optax.apply_updates(system, optimizer.update(grads, opt_state, system)[0])
```

Float-array fields are trainable; everything else is fixed at model construction. Keeping `y0`
as a field is how the initial state is trained with the weights: pass it to `solve` and its
gradient comes back in the same pytree. See
[examples/neural-ode-weather-prediction](examples/neural-ode-weather-prediction) for a full
training loop, and the docs for the rules (static fields, `out`/`mass` overrides, batching by
`jax.vmap` over leaves).

## DAEs

A singular mass matrix gives a semi-explicit index-1 DAE, solved with the implicit solvers
(`bdf`, the default, `esdirk34` or `tr_bdf2`) and differentiable like any other model. Write the
algebraic equations as rows of `rhs` and mark them with `algebraic=` (an int: the last states
are algebraic; or a boolean mask):

```python
# y' = -p0 * y   and   0 = z - p1 * y
model = Model(
    rhs=lambda x, p, t: jnp.array([-p[0] * x[0], x[1] - p[1] * x[0]]),
    nstates=2,
    nparams=2,
    algebraic=1,
)
p = jnp.array([0.7, 2.0])
ys = model.solve(jnp.array([1.0, p[1]]), p, t_eval)  # a consistent y0
```

The initial state is projected onto the algebraic constraints automatically (diffsol solves the
algebraic variables at `t0`), and the gradients account for it — so the algebraic entries of the
`y0` you pass are only a starting guess. For a general (time-dependent)
mass matrix, give `mass=lambda v, p, t: ...`, the action `M(t) v`, which must be linear in `v`
and must not use `p` (the diffsol sensitivities have no `dM/dp` term, so a parameter-dependent
mass would give a wrong gradient and is rejected — fold such parameters into `rhs` instead).
Higher-index DAEs must be reduced to index 1 first.

## Building (development)

Requirements: Python ≥ 3.12, a Rust toolchain, cmake and a C++20 compiler, and a CPU PJRT plugin.

1. Build diffsol-c from a [diffsol](https://github.com/martinjrobins/diffsol) checkout:

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

3. Get a CPU PJRT plugin. jaxlib does not ship one; a prebuilt plugin for linux/amd64 is inside
   [gopjrt](https://github.com/gomlx/gopjrt)'s `gomlx_xlabuilder_linux_amd64.tar.gz` release
   asset (`lib/gomlx/pjrt/pjrt_c_api_cpu_plugin.so`), or build `pjrt_c_api_cpu_plugin.so` from
   [openxla/xla](https://github.com/openxla/xla). Put it in `~/.local/share/pjrt/` (found
   automatically), or point `DIFFSOL_JAX_PJRT_PLUGIN` at it.

4. Make the package importable from the venv (once per venv):

   ```sh
   echo "$PWD/python" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/diffsol_jax_dev.pth"
   ```

   After this, `uv run python` and `.venv/bin/python` import `diffsol_jax` from `python/`, with
   the native library from `build/` — no `sys.path` changes needed in scripts.

5. Run the tests:

   ```sh
   .venv/bin/python -m pytest tests
   ```

## Status

This branch is a rewrite onto the PJRT architecture (vendored from the diffsol
[`diffsol-jax2` example](https://github.com/martinjrobins/diffsol/tree/diffsol-jax2/examples/diffsol-jax),
see [issue #303](https://github.com/martinjrobins/diffsol/issues/303)). CPU and `f64` only for
now; GPU arrives with diffsol's GPU backend by loading the CUDA PJRT plugin instead.
