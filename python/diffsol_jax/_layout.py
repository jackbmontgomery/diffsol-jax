"""The pytree boundary: one pytree flattened into one vector, and back."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from ._types import F64, DiffsolJaxError


def is_array(x) -> bool:
    """An array, or a shape and dtype standing in for one (`jax.eval_shape` of a network)."""
    return isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct))


def _is_param(leaf) -> bool:
    """A leaf is a value if it is an array with a float (or complex) dtype; every other leaf is
    captured by the equations as a constant."""
    return is_array(leaf) and jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)


def _template(spec, what):
    """The template a layout is built from: an int `n` is shorthand for one `f64[n]` leaf."""
    if isinstance(spec, bool):
        raise DiffsolJaxError(f"{what} must be an int or a pytree, not a bool")
    if isinstance(spec, (int, np.integer)):
        if spec < 0:
            raise DiffsolJaxError(f"{what} must not be negative, got {spec}")
        return jax.ShapeDtypeStruct((int(spec),), F64)
    return spec


class Layout:
    """One pytree flattened into one `f64[n]` vector.

    The float-array leaves of the template are the values, in flatten order, and carry only
    their shape and dtype. Any other leaf is a constant: `constants=False` (the state and the
    output, which are integrated) rejects it, and otherwise (the parameters) its value is taken
    from the template and compiled into the equations."""

    def __init__(self, spec, what: str, *, constants: bool = True):
        self.what = what
        template = _template(spec, what)
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(template)
        self.paths = [jax.tree_util.keystr(path) for path, _ in leaves]
        self.is_param = [_is_param(leaf) for _, leaf in leaves]
        self.shapes = [
            jnp.shape(leaf) if p else None
            for p, (_, leaf) in zip(self.is_param, leaves)
        ]
        self.constants = []
        for p, path, (_, leaf) in zip(self.is_param, self.paths, leaves):
            if p:
                self.constants.append(None)
            elif not constants:
                raise DiffsolJaxError(
                    f"{what}{path} is not a float array, and {what} is integrated, so every "
                    f"leaf of it must be one. Got {leaf!r}"
                )
            elif isinstance(leaf, jax.ShapeDtypeStruct):
                raise DiffsolJaxError(
                    f"{what}{path} has the dtype {leaf.dtype}, which is not differentiable, so "
                    "the equations capture its value -- but a ShapeDtypeStruct has no value. "
                    "Give the value itself, or a float dtype to make it a parameter"
                )
            else:
                self.constants.append(leaf)
        self.n = sum(math.prod(s) for s in self.shapes if s is not None)

    def __repr__(self):
        one = len(self.shapes) == 1 and self.is_param[0]
        return f"{self.what}={self.n}" if one else f"{self.what}=<{self.n} in a pytree>"

    def unravel(self, flat):
        """The pytree rebuilt from the flat `f64[n]` vector, with the constants put back."""
        leaves, i = [], 0
        for param, shape, constant in zip(self.is_param, self.shapes, self.constants):
            if not param:
                leaves.append(constant)
                continue
            assert shape is not None
            k = math.prod(shape)
            leaves.append(flat[i : i + k].reshape(shape))
            i += k
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def ravel(self, tree, what=None, hint=None) -> jax.Array:
        """The value leaves of `tree` as one `f64[n]` vector, checked against the template."""
        what = what or self.what
        tail = f": {hint}" if hint else ""
        leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
        if treedef != self.treedef:
            raise DiffsolJaxError(
                f"{what} does not match the structure the problem declares: got {treedef}, "
                f"expected {self.treedef}{tail}"
            )
        flat = []
        for param, shape, path, (_, leaf) in zip(
            self.is_param, self.shapes, self.paths, leaves
        ):
            if not param:
                if _is_param(leaf):
                    raise DiffsolJaxError(
                        f"{what}{path} is an inexact array, but the matching leaf of the "
                        "template was not, so the equations captured it as a constant. Declare "
                        "a float leaf there to make it a parameter, or pass the value the "
                        "problem was built with"
                    )
                continue
            leaf = jnp.asarray(leaf)
            if leaf.shape != shape:
                raise DiffsolJaxError(
                    f"{what}{path} has the shape {leaf.shape}, expected {shape}{tail}"
                )
            flat.append(leaf.ravel().astype(F64))
        return jnp.concatenate(flat) if flat else jnp.zeros((0,), F64)

    def unpack(self, buffer: jax.Array):
        """The pytree of a trailing `[..., n]` axis of `buffer`, each leaf `[..., *shape]`."""
        leaves, i = [], 0
        for shape in self.shapes:
            assert shape is not None
            k = math.prod(shape)
            leaves.append(buffer[..., i : i + k].reshape(buffer.shape[:-1] + shape))
            i += k
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def mask(self, spec) -> list[bool]:
        """A boolean mask over the flat vector: an int `k` for its last `k` entries, a flat
        sequence of `n` bools, or a pytree matching the template."""
        if isinstance(spec, bool):
            raise DiffsolJaxError("a mask must be an int or a boolean mask, not a bool")
        if isinstance(spec, (int, np.integer)):
            k = int(spec)
            if not 0 < k < self.n:
                raise DiffsolJaxError(
                    f"algebraic must be in [1, {self.n - 1}], got {k}"
                )
            return [False] * (self.n - k) + [True] * k
        if (
            isinstance(spec, (list, tuple, jax.Array, np.ndarray))
            and len(spec) == self.n
        ):
            return [bool(a) for a in spec]
        leaves, treedef = jax.tree_util.tree_flatten(spec)
        if treedef != self.treedef:
            if isinstance(spec, (list, tuple, jax.Array, np.ndarray)):
                raise DiffsolJaxError(
                    f"a flat boolean mask must have length {self.n}, got {len(spec)}"
                )
            raise DiffsolJaxError(
                f"a mask must be an int, a sequence of {self.n} bools, or a pytree matching "
                f"{self.what}: got {treedef}, expected {self.treedef}"
            )
        mask = []
        for leaf, shape in zip(leaves, self.shapes):
            assert shape is not None
            mask += np.broadcast_to(np.asarray(leaf, bool), shape).ravel().tolist()
        return mask
