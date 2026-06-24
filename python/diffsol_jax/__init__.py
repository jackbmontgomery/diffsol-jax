r"""JAX bindings for the diffsol ODE solver.

A user writes the ODE right-hand side as a plain Python function ``rhs(t, y, p)``.
``ODEProblem`` lowers it to diffsl, JIT-compiles it with diffsol's Cranelift backend,
and exposes a ``ODEProblem.solve`` method that is compatible with
``jax.jit``, ``jax.grad``, ``jax.jvp``/``jax.jacfwd``, and ``jax.vmap``.
"""

from .problem import ODEProblem
from .solver_type import OdeSolverType

__all__ = ["ODEProblem", "OdeSolverType"]
