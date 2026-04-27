# diffsol-jax

JAX wrapper around [diffsol](https://github.com/martinjrobins/diffsol), a Rust ODE solver library.

diffsol-jax exposes diffsol's ODE solvers via `jax.ffi` so they can be called inside `jax.jit` and
differentiated with `jax.grad`. You write a right-hand-side function in Python; the library lowers
it to a [DiffSL](https://martinjrobins.github.io/diffsl/) source string, compiles it once, and
caches the result — subsequent calls skip recompilation and update parameters in place. Gradients
are computed via diffsol's discrete adjoint, wrapped with `jax.custom_vjp`.

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from diffsol_jax import ODEProblem

def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)

params = jnp.array([1.5, 1.0, 0.75, 3.0])
y0     = jnp.array([1.0, 0.5])

problem = ODEProblem(lotka_volterra, y0=y0, params=params)

t_span = jnp.array([0.0, 10.0])
ts, ys = jax.jit(lambda p: problem.solve(p, t_span))(params)
# ts: float64[200],  ys: float64[200, 2]
```

---

[Getting started](getting-started.md) — install and first solve. [API reference](api/index.md) —
full public API. [Benchmarks](benchmarks.md) — forward and gradient timing vs diffrax.
