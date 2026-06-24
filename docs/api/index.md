# API overview

The main entry point is `ODEProblem`. Construct it with your RHS function, an initial state, and an
example parameter vector; it compiles the problem once and exposes a `solve` method that is
compatible with `jax.jit`, `jax.grad`, `jax.jvp`/`jax.jacfwd`, `jax.vjp`, and `jax.vmap`.

## Public symbols

| Symbol                                                      | Kind  | Description                                       |
| ----------------------------------------------------------- | ----- | ------------------------------------------------- |
| [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem)       | class | Compiled ODE problem; exposes the `solve` method. |
| [`OdeSolverType`](diffsol_jax.md#diffsol_jax.OdeSolverType) | enum  | Integration methods; accepts the name string too. |

## Solving

`solve` takes the parameters and the output times, plus optional solver and tolerance settings:

```python
problem = ODEProblem(rhs, y0=y0, params=params)
ts, ys = problem.solve(params, t_eval, ode_solver="tsit45", rtol=1e-5, atol=1e-6)
```

| Argument     | Default             | Description                                             |
| ------------ | ------------------- | ------------------------------------------------------- |
| `params`     | -                   | Parameter vector, shape `(n_params,)`.                  |
| `t_eval`     | -                   | Output times, shape `(n_times,)`.                       |
| `ode_solver` | `OdeSolverType.BDF` | An `OdeSolverType` or its case-insensitive name string. |
| `rtol`       | `1e-5`              | Relative tolerance.                                     |
| `atol`       | `1e-6`              | Absolute tolerance.                                     |

Returns `(ts, ys)`: `ts` of shape `(n_times,)` (equal to `t_eval`) and `ys` of shape
`(n_times, n_state)`.

## Solvers

`ode_solver=` selects the integration method. Pass an `OdeSolverType` member or its name string.

| `ode_solver=`     | `OdeSolverType`          | Type                           |
| ----------------- | ------------------------ | ------------------------------ |
| `"bdf"` (default) | `OdeSolverType.BDF`      | BDF (implicit, variable-order) |
| `"tsit45"`        | `OdeSolverType.TSIT45`   | Tsitouras 4(5) (explicit)      |
| `"esdirk34"`      | `OdeSolverType.ESDIRK34` | ESDIRK3(4) (implicit)          |
| `"tr_bdf2"`       | `OdeSolverType.TR_BDF2`  | TR-BDF2 (implicit)             |

The same solver is used for the forward solve and for the augmented forward-sensitivity solve that
backs `jax.grad`/`jax.jvp`/`jax.vjp` — there is no separate adjoint pass. See
[Computing gradients](../gradients.md) for the maths.
