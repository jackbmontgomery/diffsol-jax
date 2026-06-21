# API overview

The main entry point is `ODEProblem`. Construct it with your RHS function, an initial state, and an
example parameter vector; it compiles the problem once and exposes a `solve` callable that is
compatible with `jax.jit` and `jax.grad`.

## Public symbols

| Symbol                                                | Kind       | Description                                                       |
| ----------------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem) | class      | Compiled ODE problem; exposes a `solve` method.                   |
| [`ODE_RHS`](diffsol_jax.md#diffsol_jax.ODE_RHS)       | type alias | `Callable` — type annotation for RHS functions.                   |
| [`ODE_SOLVER`](diffsol_jax.md#diffsol_jax.ODE_SOLVER) | type alias | `Literal["bdf", "tsit45", "esdirk34", "tr_bdf2"]` — solver names. |

## Solvers

Pass the `ode_solver=` argument to `solve` to select a solver.

| `ode_solver=`     | Type                           |
| ----------------- | ------------------------------ |
| `"bdf"` (default) | BDF (implicit, variable-order) |
| `"tsit45"`        | Tsitouras 4(5) (explicit)      |
| `"esdirk34"`      | ESDIRK3(4) (implicit)          |
| `"tr_bdf2"`       | TR-BDF2 (implicit)             |

The same solver is used for the forward solve and for the augmented forward-sensitivity solve that
backs `jax.grad`/`jax.jvp`/`jax.vjp` -- there is no separate adjoint pass.

```python
problem = ODEProblem(rhs, y0=y0, params=params)
problem.solve(params, t_span, ode_solver="tsit45")
```
