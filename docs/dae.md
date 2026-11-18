# DAEs

diffsol-jax solves differential-algebraic equations of the form

$$ M(t)\, \frac{\mathrm{d}u}{\mathrm{d}t} = f(u, p, t), $$

where the mass matrix $M$ may be **singular**: rows of $M$ that are zero turn
the corresponding rows of $f$ into algebraic constraints $0 = f_i(u, p, t)$.
This is the semi-explicit, **index-1** class of DAEs. Pure-JAX solvers (diffrax)
do not support it.

## Marking algebraic states

Write the algebraic equations as rows of `rhs` and mark them with `algebraic=` -
an int (the last states are algebraic) or a boolean mask:

```python
# y' = -p0 * y   and   0 = z - p1 * y
problem = ODEProblem(
    rhs=lambda x, p, t: jnp.array([-p[0] * x[0], x[1] - p[1] * x[0]]),
    nstates=2,
    params=2,
    algebraic=1,  # or algebraic=[False, True]
)

p = jnp.array([0.7, 2.0])
ys = solve(problem, jnp.array([1.0, p[1]]), t_eval, p)  # a consistent y0
```

A DAE needs one of the implicit solvers (`solver="bdf"`, the default,
`"esdirk34"` or `"tr_bdf2"`); building a problem with the explicit `"tsit45"`
and a mass matrix is rejected.

A classic example: the Robertson system with its conservation law as a
constraint,

```python
def rhs(x, p, t):
    k1, k2, k3 = p
    r1, r2 = k1 * x[0] - k3 * x[1] * x[2], k2 * x[1] ** 2
    return jnp.array([-r1, r1 - r2, x[0] + x[1] + x[2] - 1.0])


robertson = ODEProblem(rhs, nstates=3, params=3, algebraic=1)
```

## Consistent initial conditions

The initial state must satisfy the algebraic constraints. diffsol does this
automatically: a `y0` whose algebraic entries are inconsistent is projected onto
the constraints before the solve, so those entries are only a starting guess.
The sensitivity and adjoint systems get the matching consistent initialisation,
so the gradient with respect to `y0` accounts for the projection.

## General mass matrices

For a non-diagonal or time-dependent mass, give `mass=lambda v, p, t: ...`, the
action $v \mapsto M(t)\,v$. It must be **linear in `v`** and must
**not use the parameters**.

## Limits

- **Index 1 only.** Higher-index DAEs must be reduced to index 1 first.
- **Linear in $u'$.** Fully implicit systems $F(u', u, p, t) = 0$ that cannot be
  written with a state-independent mass matrix are not supported.
