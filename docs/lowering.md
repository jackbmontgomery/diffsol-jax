# From Python to DiffSL

`ODEProblem` lets you write an ODE right-hand side as an ordinary Python
function:

```python
def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)
```

diffsol cannot run Python. It runs
[DiffSL](https://martinjrobins.github.io/diffsl/), a small tensor DSL that it
JIT-compiles to native code. So before diffsol ever sees your model, `diffsol-jax`
must translate that Python function into DiffSL source. This page explains how,
and why it is done the way it is.

## Why not parse Python?

The naive approach — walk the Python AST and emit DiffSL — is a trap. Python is a
large, dynamic language: list comprehensions, helper functions, NumPy
broadcasting, control flow, closures. An AST translator would have to understand
all of it, and would break the moment a user wrote the same maths a slightly
different way.

We sidestep the entire problem by reusing JAX's tracing machinery. The user
function is already written with `jax.numpy`, so JAX can **trace** it into a
[jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) — a typed, flattened
intermediate representation. The jaxpr is what we translate. This buys us three
things for free:

- **A tiny, fixed vocabulary.** No matter how the user writes the maths, JAX
  normalises it to a small, closed set of *primitives* (`add`, `mul`, `sin`, …).
  We only have to teach the lowerer about each primitive once.
- **Shapes and dtypes already resolved.** Every intermediate in a jaxpr has a
  known shape and dtype. We never have to infer them.
- **Single static assignment.** Each value is assigned exactly once, so the
  translation is a straight, order-preserving walk with no scoping puzzles.

## What a jaxpr looks like

`jax.make_jaxpr(lotka_volterra)(t, y, p)` produces something like:

```text
{ lambda ; a:f64[] b:f64[2] c:f64[4]. let
    d:f64[1] = slice[limit_indices=(1,) start_indices=(0,)] b
    e:f64[]  = squeeze[dimensions=(0,)] d          # y[0]
    ...
    i:f64[]  = squeeze[...] h                       # p[0]
    p:f64[]  = mul i e                              # alpha * x
    q:f64[]  = mul k e
    r:f64[]  = mul q g
    s:f64[]  = sub p r                              # alpha*x - beta*x*yy
    ...
  in (s, w) }
```

A jaxpr is just a list of equations, each of the form

```text
out_vars = primitive[static_params] in_vars
```

plus a list of input variables (`a`, `b`, `c` — here `t`, `y`, `p`) and a tuple
of output variables (`(s, w)` — the two components of dudt).

## The mapping

Both jaxprs and DiffSL are single-assignment tensor languages, so the spine of
the translation is almost one-to-one: **each jaxpr equation becomes one DiffSL
tensor definition.**

| jaxpr concept            | DiffSL concept                  | Example                |
| ------------------------ | ------------------------------- | ---------------------- |
| input variable `t`/`y`/`p` | `in` / `u` blocks + symbols   | `in_i { p0 = 1.5, … }` |
| equation `mul a b`       | a named tensor definition       | `v0 { p0 * y0 }`       |
| `Literal`                | an inline numeric constant      | `2.0`                  |
| output variables         | the `F` (dudt) block            | `F_i { v3, v7 }`       |

For Lotka–Volterra the whole result is:

```text
in_i { p0 = 1.5, p1 = 1.0, p2 = 0.75, p3 = 3.0 }
u_i  { y0 = 1.0, y1 = 0.5 }
v0 { p0 * y0 }
v1 { p1 * y0 }
v2 { v1 * y1 }
v3 { v0 - v2 }
v4 { p2 * y0 }
v5 { v4 * y1 }
v6 { p3 * y1 }
v7 { v5 - v6 }
F_i {
  v3,
  v7,
}
```

Two details stop this from being literally one line of code per equation.

### Subscripts

DiffSL is a tensor language: a rank-1 value is written `name_i`, a scalar is just
`name`. So every value the lowerer tracks carries its **index letters** — `""`
for a scalar, `"i"` for a vector — and renders the correct form (`name` vs
`name_i`) wherever it is referenced. This is the `Value` type in the
implementation.

### Tracer noise

Tracing introduces structural primitives that carry no arithmetic. The lowerer
recognises each one and, in most cases, *rebinds a name* rather than emitting
code — keeping the output close to what a human would write by hand:

- **`slice` + `squeeze`** — the trace of `p[0]` / `y[1]`. Rather than emit array
  indexing, the lowerer maps these straight to the parameter/state symbol
  (`p0`, `y1`).
- **`broadcast_in_dim` + `concatenate`** — the trace of `jnp.array([a, b])`. A
  scalar that is broadcast to a length-1 vector purely to be concatenated is
  *elided*: the scalar is spliced directly into the concatenation, so
  `jnp.array([f, g])` becomes a single `cat_i { f, g }` instead of two dead
  intermediate vectors.
- **`pjit` / `closed_call`** — JAX's function-call wrappers. DiffSL has no call
  construct, so these subjaxprs are **inlined**: bind the callee's arguments,
  emit its body, alias its results.
- **`convert_element_type`** — dtype casts (everything is f64), elided.

## Architecture

The translator lives in
[`diffsol_jax/lowering.py`](api/diffsol_jax.md). The core is a `Lowering` object
that holds all translation state:

- a **symbol table** mapping each jaxpr variable to a `Value`,
- a fresh-name counter,
- the accumulated DiffSL body lines.

The walk is a **dispatch table** keyed on primitive name. Each handler consumes
one equation and either emits a line or rebinds names. Anything not in the table
falls through to the generic elementwise handler (arithmetic and math
functions). Critically, a primitive the lowerer has *not* been taught about
raises `NotImplementedError` naming the offending primitive — it never silently
miscompiles.

```text
jax.make_jaxpr(rhs)        # Python -> jaxpr (JAX does this)
        │
        ▼
Lowering.lower_eqns(...)   # jaxpr equations -> DiffSL body lines
        │                  #   dispatch on primitive name:
        │                  #     slice/squeeze -> p/y symbols
        │                  #     broadcast/concatenate -> tensor literals
        │                  #     pjit/closed_call -> inline
        │                  #     add/mul/sin/... -> elementwise
        ▼
in / u / body / F blocks   # assembled into final DiffSL source
```

## Adding a new primitive

If a user model hits a primitive we do not yet support, the failure is explicit:

```text
NotImplementedError: primitive 'reduce_sum' not supported
```

To add support, write a handler method on `Lowering` that maps the primitive to
its DiffSL form and register it in the `_HANDLERS` table (or extend
`_elementwise` for a simple scalar op). Because the public contract is "jaxpr in,
DiffSL out", the safest way to validate a change is to assert on the generated
source for a representative model — the translation is deterministic, so the
output is stable.

## Limitations

The lowerer targets the subset of JAX needed to express ODE right-hand sides:

- **Scalars and rank-1 vectors**, dtype **f64** only.
- **No control flow** (`cond`, `scan`, `while_loop`) — DiffSL has none.
- The RHS must return either a tuple of `n_state` scalars or a single
  length-`n_state` vector.
- `concatenate` is supported along axis 0 only.

These cover the common ODE models (Lotka–Volterra, Lorenz, chemical kinetics,
…); unsupported constructs fail loudly rather than producing a wrong solver.
