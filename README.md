<p align="center">
  <img src="assets/diffsol-jax.png" alt="Logo" style="width: 75%;">
</p>

Write JAX native equations, compile it, and solve it with
[diffsol](https://github.com/martinjrobins/diffsol). Compatible with all
functional transformations (`jax.jit`, `jax.vmap`, `jax.grad`, etc.).

## Key Features

- FFI calls are wrapped as primitive operations supporting `jacfwd` and
  `jacrev`;
- Double and single precision;
- Forward and adjoint gradient computations;
- Semi-explicit DAE support;
- Vectorized solves are solved as lanes on a single solve;
- Using PyTrees as the state;
- Support for neural differential equations;

## Sharp Bits

- **CPU only, for now.**
- **No higher-order derivatives.** `grad(grad(solve(...)))` does not work.
- **Small non-stiff performance.** Each solver callback costs ~2.5 µs of fixed
  PJRT overhead; an explicit method on a tiny system is often faster in
  `diffrax`.

## Installation

```bash
pip install "diffsol-jax[cpu]"
```

## Documentation

<https://jackbmontgomery.github.io/diffsol-jax/>

## Quick Example

```python
import jax, jax.numpy as jnp
from diffsol_jax import ODEProblem, solve

def rhs(y, p, t):
   return p[0] * y * (1.0 - y)

N = 50
y0, p = jnp.array([0.1]), jnp.array([1.1])
t0, t1 = 0.0, 5.0
ts = jnp.linspace(t0, t1, N)

problem = ODEProblem(rhs, state=1, params=1)
ys = solve(problem, y0, p, t0, t1, saveat=ts)  # [N, nstates]
```

## See Also: Other Libraries in the JAX Ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and
everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for
shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam,...)
optimisers.  
[paramax](https://github.com/danielward27/paramax): parameterizations and
constraints for PyTrees.  

**Scientific computing**  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding,
minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian
sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX
conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX
honourable mention!)  
[diffrax](https://github.com/patrick-kidger/diffrax): numerical differential
equation solvers.

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other
JAX projects.  
