import abc
import dataclasses
import math
from typing import Any, Self, cast, dataclass_transform

import jax
import jax.numpy as jnp
import numpy as np

from ._lib import DiffsolJaxError


@dataclass_transform(frozen_default=True, field_specifiers=(dataclasses.field,))
class System(abc.ABC):
    """An ODE system: the differentiable inputs as fields, the equations as methods.

    Subclasses are frozen dataclasses registered as pytrees. Field leaves that are float (or
    complex) arrays are trainable; every other field is fixed when the model is built."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        datacls: Any = dataclasses.dataclass(frozen=True)(cls)
        fields = dataclasses.fields(datacls)
        static = [f.name for f in fields if f.metadata.get("static", False)]
        data = [f.name for f in fields if not f.metadata.get("static", False)]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=static)

    def replace(self, **changes: Any) -> Self:
        """A copy with the given fields replaced."""
        return dataclasses.replace(cast(Any, self), **changes)

    @abc.abstractmethod
    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """The right-hand side `[nstates]` at state `x [nstates]` and time `t`."""

    def out(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Override: the output `[nout]` at state `x` and time `t`."""
        raise NotImplementedError

    def mass(self, v: jax.Array, t: jax.Array) -> jax.Array:
        """Override: the mass matrix action `M(t) v`, linear in `v` and not using the
        trainable fields."""
        raise NotImplementedError


def _is_trainable(leaf):
    return isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
        jnp.result_type(leaf), jnp.inexact
    )


class ParamSpec:
    def __init__(self, variables_like):
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(variables_like)
        self.paths = [jax.tree_util.keystr(path) for path, _ in leaves]
        self.trainable = [_is_trainable(leaf) for _, leaf in leaves]
        self.statics = [
            None if t else leaf for t, (_, leaf) in zip(self.trainable, leaves)
        ]
        self.shapes = [
            jnp.shape(leaf) if t else None
            for t, (_, leaf) in zip(self.trainable, leaves)
        ]
        self.nparams = sum(math.prod(s) for s in self.shapes if s is not None)
        if self.nparams == 0:
            raise DiffsolJaxError(
                "variables_like has no trainable leaves: at least one leaf must be an "
                "array with a float (or complex) dtype"
            )

    def ravel(self, variables):
        """The trainable leaves of `variables` as one `f64[nparams]` vector."""
        leaves, treedef = jax.tree_util.tree_flatten_with_path(variables)
        if treedef != self.treedef:
            raise DiffsolJaxError(
                f"variables does not match variables_like: got {treedef}, expected "
                f"{self.treedef}"
            )
        flat = []
        for t, shape, path, (_, leaf) in zip(
            self.trainable, self.shapes, self.paths, leaves
        ):
            if not t:
                if _is_trainable(leaf):
                    raise DiffsolJaxError(
                        f"variables{path} is an inexact array, but the matching leaf of "
                        "variables_like was not, so the model treats it as fixed. Build "
                        "the model with it trainable in variables_like, or pass the "
                        "construction-time value"
                    )
                continue
            leaf = jnp.asarray(leaf)
            if leaf.shape != shape:
                raise DiffsolJaxError(
                    f"variables{path} has shape {leaf.shape}, expected {shape} (batch "
                    "with jax.vmap, not with extra axes)"
                )
            flat.append(leaf.ravel().astype(jnp.float64))
        return jnp.concatenate(flat)

    def bind_equations(self, rhs, out, mass):
        """The user functions over the pytree, as functions of the flat vector."""

        def wrap(f):
            if f is None:
                return None
            return lambda x, p, t: jnp.asarray(f(x, self.bind(p), t), jnp.float64)

        return wrap(rhs), wrap(out), wrap(mass)

    def bind(self, flat):
        """The pytree rebuilt from the flat `f64[nparams]` vector."""
        leaves, i = [], 0
        for t, shape, static in zip(self.trainable, self.shapes, self.statics):
            if t:
                assert shape is not None
                n = math.prod(shape)
                leaves.append(flat[i : i + n].reshape(shape))
                i += n
            else:
                leaves.append(static)
        return jax.tree_util.tree_unflatten(self.treedef, leaves)
