# API overview

The main entry point is `Model`. Construct it with your equation functions and sizes; it exposes
a `solve` attribute that is compatible with `jax.jit`, `jax.grad`, `jax.jacrev`, `jax.jvp`,
`jax.jacfwd`, and `jax.vmap`.

## Public symbols

| Symbol                                                          | Kind      | Description                                   |
| --------------------------------------------------------------- | --------- | --------------------------------------------- |
| [`Model`](diffsol_jax.md#diffsol_jax.Model)                     | class     | A compiled model; exposes `solve`.            |
| [`System`](diffsol_jax.md#diffsol_jax.System)                   | class     | Abstract base: variables as fields, equations as methods — see [Systems](../getting-started.md#systems-classes-instead-of-flat-vectors). |
| [`DiffsolJaxError`](diffsol_jax.md#diffsol_jax.DiffsolJaxError) | exception | Raised on configuration and solver errors.    |

## The equations

The recommended form is a `System`: `Model(system, **options)` derives the equations from the
methods `init(self, t)`, `__call__(self, x, t)`, and the optional overrides `out(self, x, t)`
and `mass(self, v, t)`, with `nstates`/`nout` inferred by shape tracing.

In the functional form, all functions act on **one lane**: `x` has shape `[nstates]`, `p` has
shape `[nparams]`, and `t` is a scalar. Any traceable JAX code is allowed. With
`variables_like=` instead of `nparams`, `p` is instead the variables pytree (an `eqx.Module`,
a dict, ...).

| Function        | Signature      | Required | Description                                             |
| --------------- | -------------- | -------- | ------------------------------------------------------- |
| `rhs`           | `(x, p, t)`    | yes      | The right-hand side, returning `[nstates]`.             |
| `init`          | `(p, t)`       | yes      | The initial state, returning `[nstates]`.               |
| `out`           | `(x, p, t)`    | no       | An output function returning `[nout]`; `ys` then has `nout` columns. |
| `mass`          | `(v, p, t)`    | no       | The mass matrix action `M(t) v`, linear in `v`, not using `p`. See [DAEs](../dae.md). |

## Solving

```python
model = Model(system, ode_solver="bdf", rtol=1e-6, atol=1e-6)        # from a System
model = Model(rhs, init, nstates=2, nparams=4)                       # functional form
ys = model.solve(params, t_eval)
```

`solve(params, t_eval)` takes `params` of shape `[..., nparams]` (leading axes are solved as
lanes of one batched solve) and `t_eval` of shape `[nt]`, and returns `ys` of shape
`[..., nt, nout]` (`nout = nstates` if the model has no `out`). For a `System` model (or
`variables_like=`), `solve(system, t_eval)` takes the pytree instead — the current field
values, e.g. mid-training — and batching is by `jax.vmap` over its leaves.

## Solvers

| `ode_solver=`     | Type                             | Mass matrices / DAEs |
| ----------------- | -------------------------------- | -------------------- |
| `"bdf"` (default) | BDF (implicit, variable-order)   | yes                  |
| `"esdirk34"`      | ESDIRK3(4) (implicit)            | yes                  |
| `"tr_bdf2"`       | TR-BDF2 (implicit)               | yes                  |
| `"tsit45"`        | Tsitouras 4(5) (explicit)        | no                   |

Reverse mode uses the diffsol adjoint (`ad="adjoint"`, the default) or the transpose of the
forward sensitivities (`ad="forward"`); forward mode always uses forward sensitivities. See
[Computing gradients](../gradients.md).
