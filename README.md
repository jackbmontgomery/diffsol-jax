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
    rhs=lambda x, p, t: p[0] * x * (1.0 - x),   # one lane: x [nstates], p [nparams]
    init=lambda p, t: jnp.array([0.1]),
    nstates=1, nparams=1,
)
t = jnp.linspace(0.0, 5.0, 11)
ys = model.solve(jnp.array([1.0]), t)                        # [nt, nstates]
ys = jax.vmap(model.solve, (0, None))(jnp.ones((64, 1)), t)  # one solve with 64 lanes
g = jax.grad(lambda p: model.solve(p, t).sum())(jnp.array([1.0]))
```

Reverse mode (`ad="adjoint"`, the default) uses the diffsol adjoint with checkpointing; forward
mode (`ad="forward"`) uses diffsol forward sensitivities. A `jax.vmap` over `params` becomes more
lanes in a single batched solve, not a loop of solves.

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
   [openxla/xla](https://github.com/openxla/xla).

4. Run the tests:

   ```sh
   export DIFFSOL_JAX_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
   .venv/bin/python -m pytest tests
   ```

## Status

This branch is a rewrite onto the PJRT architecture (vendored from the diffsol
[`diffsol-jax2` example](https://github.com/martinjrobins/diffsol/tree/diffsol-jax2/examples/diffsol-jax),
see [issue #303](https://github.com/martinjrobins/diffsol/issues/303)). CPU and `f64` only for
now; GPU arrives with diffsol's GPU backend by loading the CUDA PJRT plugin instead.
