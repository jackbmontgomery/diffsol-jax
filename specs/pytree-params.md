# Spec: pytree variables

Status: implemented on this branch, 2026-11-16

## Problem

`Model.solve(params, t_eval)` takes a flat `f64[..., nparams]` vector, and the user functions
receive that same flat vector: `rhs(x, p, t)` with `p[nparams]`. For classical ODEs with a
handful of named rate constants this is fine. For the main upcoming use case — ML, where the
parameters are an Equinox module (or any pytree) updated by an optimizer — the flat vector
pushes bookkeeping onto every user:

```python
# today: examples/neural-ode-weather-prediction/main.py
nn_params, static = eqx.partition(neural_net, eqx.is_array)
nn_p0, unravel_nn_params = ravel_pytree(nn_params)

def rhs(x, p, _t):
    params = unravel_nn_params(p[DATA_DIM:])      # slice by hand-computed offset
    net = eqx.combine(params, static)             # rebuild the module
    return net(x)

model = Model(rhs=rhs, init=lambda p, _t: p[:DATA_DIM],
              nstates=DATA_DIM, nparams=DATA_DIM + nn_p0.shape[0], ...)

def predict(nn_p, t_eval):
    full_p = jnp.concatenate([y0, nn_p])          # pack two parameter groups
    return model.solve(full_p, t_eval)
```

Four separate chores: partition the module, ravel it, compute `nparams` and the slice offsets,
and concatenate heterogeneous parameter groups into one vector. The optimizer then runs on the
flat vector, so the natural optax/equinox idiom (per-leaf updates, `eqx.apply_updates`,
per-module learning rates) is lost too.

## Proposal

Let the differentiable inputs of a solve be **any pytree**, declared once at model
construction with an example (`variables_like=`, replacing `nparams`), and passed as-is to
`solve`. The user functions receive the pytree; gradients come back as a pytree of the same
structure, so optax works on it directly.

The recommended form is the exported abstract `System` class — a frozen dataclass registered
as a pytree whose **fields are the variables** and whose **methods are the equations**, so
that "equations + variables" is the visible design of the library. `Model(system)` derives
everything from it:

```python
class WeatherSystem(System):
    y0: jax.Array                                  # the initial state, trained too
    net: NeuralNetwork                             # an eqx.Module — already a pytree

    def init(self, t):                             # required
        return self.y0

    def __call__(self, x, t):                      # required: the rhs
        return self.net(x)


system = WeatherSystem(y0=ys[0], net=NeuralNetwork(data_dim=4, key=key))
model = Model(system, ode_solver="tsit45")         # nstates inferred from init

ys = model.solve(system, t_eval)                   # [nt, nout]
loss = lambda s: jnp.sum((model.solve(s, t_eval) - data) ** 2)
grads = jax.grad(loss)(system)                     # a WeatherSystem of gradients
updates, opt_state = opt.update(grads, opt_state, system)
system = optax.apply_updates(system, updates)      # no ravel/unravel anywhere
```

`System` enforces the two **methods** (`init(t)` and `__call__(x, t)`, via `abc`), not any
field names: the fields are the user's variables, named freely. A trainable initial state is
the one-field pattern above; a fixed one is `def init(self, t): return Y0`. The optional
overrides `out(self, x, t)` (output function, `nout` inferred) and `mass(self, v, t)` (mass
matrix action — it must not use the trainable fields, which the construction-time linearity
check enforces) complete the equation set; `algebraic=` stays a `Model` kwarg and combines
with a `System`. `nstates`/`nout` are inferred with `jax.eval_shape` on `init`/`out`.

Beyond that, `System` provides only `__init_subclass__` (dataclass +
`jax.tree_util.register_dataclass`, with `dataclasses.field(metadata={"static": True})` for
fields kept out of the leaves) and `replace`. Nothing else — filtering, serialization, and
richer module features stay equinox's job; the library only needs the pytree protocol.

Two lower-level forms remain: `Model(rhs=..., init=..., nstates=..., variables_like=pytree)`
takes the variables as **any pytree** (an `eqx.Module`, a dict, ...) with functional
equations receiving the pytree, and `nparams=<int>` stays as the flat case, unchanged.
Exactly one of a `System`, `variables_like`, or `nparams` must be given.

## Design

The key observation is that nothing below `Model.__init__` has to change. The solver core
(`_lower.py`, `_primitives.py`, the C++ side) is built around a flat `f64[B, nparams]` buffer —
that is the right representation for the callbacks, the sensitivity lanes (`P = nparams`), and
the FFI. The pytree is a **thin boundary layer** in `Model`:

1. **Partition at construction.** Split the leaves of `variables_like` into *trainable* —
   inexact-dtype arrays (`jnp.issubdtype(dtype, jnp.inexact)` after `jnp.asarray`) — and
   *static* — everything else: ints, bools, strings, activation callables, the treedef itself.
   This is `eqx.partition(…, eqx.is_inexact_array)` in spirit, implemented with
   `jax.tree_util` only (eqx modules are ordinary pytrees; no equinox dependency).

2. **Ravel the trainable part.** `flat0, unravel = ravel_pytree(trainable)` gives
   `nparams = flat0.size` for `diffsol_jax_model_new`, and a `bind` function:

   ```python
   def bind(flat_p):      # f64[nparams] -> the user's pytree
       return combine(unravel(flat_p), static_leaves)
   ```

   The user functions are wrapped once — `rhs_flat = lambda x, pf, t: rhs(x, bind(pf), t)`,
   same for `init`, `out`, `mass` — and the wrapped functions go into `Equations`. Since the
   callbacks are *traced* to StableHLO, `bind` costs nothing at solve time: the
   slice/reshape ops fuse into the compiled callback.

3. **Ravel at the solve boundary.** `solve(params, t_eval)` validates that the treedef and the
   trainable leaf shapes/dtypes match `variables_like`, ravels the trainable leaves to
   `f64[nparams]`, and calls the existing primitive. The ravel is built from
   `reshape`/`concatenate`, so JAX AD and `vmap` compose through it for free:
   - `jax.grad` / `jax.jacrev` / `jax.jvp`: the cotangent of the ravel is the unravel — the
     gradient arrives as a pytree with the structure of `params`, with each leaf cast back to
     its own dtype (that is `ravel_pytree`'s contract). Static leaves get no gradient, which is
     what `eqx.filter_grad` users expect.
   - `jax.vmap`: vmapping any (or all) leaves batches the raveled vector, which the existing
     `batch` rule turns into solver lanes. `in_axes=None` leaves still broadcast as usual.

4. **Structure is part of the model.** Changing the pytree *structure* (a different
   architecture, an extra parameter group) changes `nparams` and therefore the compiled
   callbacks and the C++ model — it requires a new `Model`, exactly as changing `nparams` does
   today. Changing leaf *values* is the hot path and costs nothing. `solve` raises
   `DiffsolJaxError` with the two treedefs printed when the structure does not match.

### Dtypes

The solver is f64 and `jax_enable_x64` is already required. f32 leaves (common in ML inits) are
accepted: they are cast to f64 on ravel, and reverse mode returns f32 gradients for them
(`unravel`'s per-leaf `astype`, whose VJP casts back). Non-inexact array leaves (int shapes,
bool masks) are static by definition and pass through `bind` untouched.

### `mass` linearity check

`_check_mass_is_linear` draws random flat vectors. The user's `mass` is wrapped through `bind`
*before* the check runs, so the existing helper applies unchanged to the wrapped flat
function.

### Validation and errors

- `variables_like` with **zero** trainable leaves: error at construction ("the model has no
  trainable parameters; mark at least one leaf as an inexact array").
- `solve` called with a mismatched treedef or leaf shape: error naming the offending leaf path
  (`jax.tree_util.keystr`).
- `nparams` and `variables_like` both given, or neither: error at construction.

### Out of scope (unchanged)

- Batching by stacking a leading axis on the raw flat vector keeps working in `nparams` mode;
  in `variables_like` mode batching is via `jax.vmap` only — that is the JAX-idiomatic spelling
  and avoids guessing which leaf axes are batch axes.
- `t_eval`, tolerances, solver choice: untouched.
- Higher-order AD, grad w.r.t. `t_eval`: same limitations as today.

## Follow-up (separate change): infer `nstates` and `nout`

With `variables_like` in hand, `init` can be shape-traced at construction:

```python
nstates = jax.eval_shape(init, variables_like, 0.0).shape[0]
nout    = jax.eval_shape(out, jax.ShapeDtypeStruct((nstates,), f64), variables_like, 0.0).shape[0]
```

making `nstates=`/`nout=` optional. Kept out of this spec's first PR to keep the diff reviewable,
but the API is designed so it bolts on without breaking anything.

## What the example becomes

`examples/neural-ode-weather-prediction/main.py` loses all of its plumbing:

```python
class WeatherSystem(System):
    y0: jax.Array
    net: NeuralNetwork

    def init(self, t):
        return self.y0

    def __call__(self, x, t):
        return self.net(x)


system0 = WeatherSystem(y0=ys[0], net=NeuralNetwork(data_dim=DATA_DIM, key=jr.key(0)))

model = Model(system0, ode_solver="tsit45")

def loss_fn(system, ts_batch, ys_batch):
    return jnp.sum((model.solve(system, ts_batch) - ys_batch) ** 2)

value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
# optax runs on the WeatherSystem; y0 trains together with the net, or freeze it
# with optax.masked / eqx.tree_at — standard tools, nothing diffsol-specific.
```

`ravel_pytree`, `eqx.partition`, `unravel_nn_params`, `DATA_DIM + nn_p0.shape[0]`, and the
`jnp.concatenate` in `predict` all disappear.

## Alternatives considered

- **diffrax-style `args=` at solve time** (`solve(params, t_eval)` vs `solve(t_eval,
  args=params)`): identical power; keeping `params` positional preserves today's signature and
  reads better under `jax.grad`.
- **Capture the module in the closure, pass only arrays** (`eqx.filter_*` style, where `Model`
  takes the module itself as `rhs`): saves one lambda in the simplest case but hides where the
  trainable state lives, and makes multi-group parameters (y0 + net) awkward again. The
  explicit pytree subsumes it: `rhs=lambda x, p, t: p["net"](x)`.
- **Ship only a helper** (`diffsol_jax.ravel(params) -> (flat, unravel)`): smallest diff, but
  it fixes none of the `solve`/`grad`/optimizer ergonomics, which is the actual complaint.

## Implementation

- `python/diffsol_jax/_params.py`: the abstract `System` base class (`abc` + dataclass +
  `register_dataclass`), and `ParamSpec` — partition, `ravel` (with treedef/shape validation
  and key-path errors), `bind`, `bind_equations`.
- `Model.__init__`: a `System` first argument derives `rhs`/`init`/`out`/`mass` from the
  methods, infers `nstates`/`nout` with `jax.eval_shape`, and rejects redundant arguments;
  otherwise it accepts `variables_like` or `nparams` (exactly one mode), wraps the user
  functions through `bind` before building `Equations`, and `check()` in `_make_solve` ravels
  instead of shape-checking in pytree mode.
- `_lower.py`, `_primitives.py`, C++: **no changes**.
- Tests: `tests/test_system.py` — system solve == flat solve with `nstates`/`nout` inference;
  adjoint/forward grad structure and dtypes (f32 leaf → f32 grad); `jacfwd`; vmap over a
  leaf; an `out` method; dict `variables_like` with a static callable leaf; static-metadata
  fields and `replace`; abstract-method enforcement; mode and solve-time mismatch errors;
  DAEs via `algebraic=` and via a `mass` method, and the trainable-`mass` rejection.
- Docs: the "Systems" section in `docs/getting-started.md`, `docs/api/index.md`, the module
  docstring, and the rewritten neural-ODE example.
