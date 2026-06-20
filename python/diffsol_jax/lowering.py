"""Lower a traced JAX function (a *jaxpr*) to a DiffSL source string.

Design
======

``ODEProblem`` lets the user write the ODE right-hand side as an ordinary
Python function ``rhs(t, y, p)``. diffsol, however, does not consume Python —
it consumes `DiffSL <https://martinjrobins.github.io/diffsl/>`_, a small tensor
DSL that it JIT-compiles to native code. This module bridges the two.

We do *not* parse Python. Instead we let JAX do the hard part: ``jax.make_jaxpr``
traces ``rhs`` into a `jaxpr <https://docs.jax.dev/en/latest/jaxpr.html>`_ — a
typed, flattened, single-assignment IR. A jaxpr is a list of equations

    out_vars = primitive[params] in_vars

where every intermediate already has a known shape/dtype and every operation is
one of a small, fixed set of *primitives* (``add``, ``mul``, ``sin``, ...).
Translating that to DiffSL is a near-mechanical, primitive-by-primitive walk —
far more robust than pattern-matching Python ASTs.

The mapping
-----------

Both IRs are single-assignment, so the spine of the translation is trivial:
each jaxpr equation becomes one DiffSL tensor definition.

================  =========================  ============================
jaxpr concept     DiffSL concept             handled by
================  =========================  ============================
``Var``           a named tensor             :class:`Value` in :class:`Lowering.values`
``Literal``       an inline numeric constant :meth:`Lowering.resolve`
``mul``/``sin``…  ``v { a * b }`` etc.       :meth:`Lowering._elementwise`
function invars   ``in``/``u`` symbols       :func:`make_diffsl_tuple`
function outvars  the ``F`` (dudt) block     :func:`make_diffsl_tuple`
================  =========================  ============================

Two things keep this from being a one-liner per equation:

1. **Subscripts.** DiffSL is a tensor language: a rank-1 value is written
   ``name_i`` and a scalar is written ``name``. Every :class:`Value` therefore
   carries its index letters (``""`` for a scalar, ``"i"`` for a vector), and
   :meth:`Value.ref` produces the right textual form on demand.

2. **Tracer noise.** JAX emits a few structural primitives that carry no
   arithmetic — ``slice``/``squeeze`` from ``p[0]`` indexing, ``broadcast_in_dim``
   and ``concatenate`` from ``jnp.array([...])``, ``pjit`` wrappers, dtype casts.
   These are handled by dedicated, documented primitive handlers that mostly
   *rebind names* rather than emit code, so the generated DiffSL stays close to
   what a human would write by hand.

The walk is a dispatch table (:attr:`Lowering._HANDLERS`) keyed on the primitive
name, replacing what would otherwise be one large ``if/elif`` chain. Anything we
have not taught the lowerer about raises ``NotImplementedError`` with the
offending primitive named, rather than silently miscompiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.extend import core as jex_core

# DiffSL index letters. Skip 'o' (reads as zero); 't' is reserved for time.
_INDEX_LETTERS = "ijklmnpqrs"

# jaxpr primitive name -> DiffSL builtin function name.
_UNARY_FNS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "abs": "abs",
    "logistic": "sigmoid",
}

# jaxpr primitive name -> DiffSL infix operator.
# ``add_any`` is JAX's tangent-accumulation primitive (from ``jvp``/``jacfwd``); it
# is semantically identical to ``add`` and appears throughout augmented-sensitivity
# RHS code.
_BINARY_OPS = {
    "add": "+",
    "add_any": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _index_letters(rank: int) -> str:
    """Return the first ``rank`` DiffSL index letters, e.g. ``2 -> "ij"``."""
    if rank == 0:
        return ""
    if rank > len(_INDEX_LETTERS):
        raise NotImplementedError(f"rank {rank} exceeds subscript pool")
    return _INDEX_LETTERS[:rank]


def _fmt_float(x: float) -> str:
    """Format a Python float as a DiffSL literal (always with a decimal point)."""
    if x == int(x):
        return f"{int(x)}.0"
    return repr(x)


@dataclass(frozen=True)
class Value:
    """A reference to an emitted DiffSL value.

    ``subs`` holds the value's index letters: ``""`` for a scalar, ``"i"`` for a
    rank-1 tensor, ``"ij"`` for rank-2, and so on. :meth:`ref` renders the value
    as it must appear inside an expression.
    """

    name: str
    subs: str = ""

    @property
    def rank(self) -> int:
        return len(self.subs)

    def ref(self) -> str:
        """Textual reference used inside an expression body."""
        return self.name if self.subs == "" else f"{self.name}_{self.subs}"


class Lowering:
    """Walks a jaxpr and accumulates DiffSL body lines.

    Holds the whole translation state: the symbol table mapping jaxpr ``Var`` s to
    :class:`Value` s, the fresh-name counter, the accumulated body ``lines``, and
    the metadata needed to recognise indexing into the parameter/state vectors.
    """

    def __init__(
        self,
        *,
        param_names: list[str],
        state_names: list[str],
        t_var,
        y_var,
        p_var,
    ):
        self.values: dict = {}
        self.lines: list[str] = []
        self._counter = 0

        self.param_names = param_names
        self.state_names = state_names
        self.n_param = len(param_names)
        self.n_state = len(state_names)

        # The jaxpr invars for t, y and p. ``slice``/``squeeze`` against y_var or
        # p_var is the trace of a ``y[i]``/``p[i]`` index and is turned straight
        # into the corresponding DiffSL symbol instead of emitting code.
        self.t_var = t_var
        self.y_var = y_var
        self.p_var = p_var

        # Scalar-broadcast elision (see _broadcast_in_dim / _concatenate):
        #   scalar_wraps:  broadcast-out var -> underlying scalar name
        #   concat_only:   broadcast-out vars whose *only* consumer is concatenate
        self.scalar_wraps: dict = {}
        self.concat_only: set = set()

        # Vector "views": maps a jaxpr var to a contiguous slice of an underlying
        # named vector, so width-1 indexing resolves to a scalar symbol/literal and
        # width-k indexing yields a narrower view. The state/parameter vectors are
        # the root views; rank-1 constvars (e.g. the one-hot tangents from
        # ``jax.jvp`` in augmented-sensitivity code) register as ``"const"`` views.
        # Each entry is ``("state"|"param", offset)`` or ``("const", offset, values)``.
        self.views: dict = {}

    # ── symbol table ─────────────────────────────────────────────────────────

    def fresh(self, hint: str = "v") -> str:
        name = f"{hint}{self._counter}"
        self._counter += 1
        return name

    def bind(self, var, value: Value) -> None:
        self.values[var] = value

    def resolve(self, atom) -> Value:
        """Resolve a jaxpr atom (``Var`` or ``Literal``) to a :class:`Value`.

        A literal becomes an anonymous scalar whose "name" is its numeric text,
        so :meth:`Value.ref` splices it inline.
        """
        if isinstance(atom, jex_core.Literal):
            val = atom.val
            if hasattr(val, "shape") and val.shape != ():
                raise NotImplementedError("non-scalar inline literal not supported")
            return Value(_fmt_float(float(val)))
        return self.values[atom]

    def emit(self, name: str, subs: str, expr: str) -> Value:
        """Emit ``name_subs { expr }`` and return the resulting :class:`Value`."""
        ref = name if subs == "" else f"{name}_{subs}"
        self.lines.append(f"{ref} {{ {expr} }}")
        return Value(name, subs)

    # ── driver ───────────────────────────────────────────────────────────────

    def lower_eqns(self, eqns) -> None:
        for eqn in eqns:
            self._HANDLERS.get(eqn.primitive.name, Lowering._elementwise)(self, eqn)

    def bind_const(self, var, val) -> None:
        """Bind a jaxpr constvar.

        Scalars become a named DiffSL scalar definition. Rank-1 constants (e.g.
        the one-hot tangents from ``jax.jvp``) are recorded in
        :attr:`const_vectors` so that width-1 ``slice``/``squeeze`` against them
        resolves to the inline literal; the vector itself is never emitted.
        """
        shape = getattr(val, "shape", ())
        if shape == ():
            name = self.fresh("c")
            self.bind(var, self.emit(name, "", _fmt_float(float(val))))
            return
        if len(shape) == 1:
            self.views[var] = ("const", 0, [float(x) for x in val])
            return
        raise NotImplementedError(
            f"constvar rank {len(shape)} (shape={shape}) not supported; "
            "only scalars and 1-D constants"
        )

    # ── primitive handlers ───────────────────────────────────────────────────
    #
    # Each handler consumes one equation: it either emits a DiffSL line (via
    # ``emit``) or merely rebinds names in the symbol table. Registered in
    # ``_HANDLERS`` below; everything not listed falls through to
    # ``_elementwise``.

    def _view_element(self, view, i: int) -> Value:
        """Resolve element ``i`` of a vector view to its DiffSL symbol/literal."""
        kind, off = view[0], view[1]
        if kind == "state":
            return Value(self.state_names[off + i])
        if kind == "param":
            return Value(self.param_names[off + i])
        return Value(_fmt_float(view[2][off + i]))  # "const"

    def _slice(self, eqn) -> None:
        """Index/sub-slice a vector view.

        A width-1 slice of the state/parameter/const vector resolves to the
        corresponding scalar symbol or literal; a width-k slice yields a narrower
        view (offset shifted) that later slices resolve through.
        """
        in_atom = eqn.invars[0]
        start = eqn.params["start_indices"][0]
        width = eqn.params["limit_indices"][0] - start
        strides = eqn.params.get("strides")
        view = self.views.get(in_atom)
        if view is not None and (strides is None or tuple(strides) == (1,)):
            if width == 1:
                self.bind(eqn.outvars[0], self._view_element(view, start))
                return
            if view[0] == "const":
                self.views[eqn.outvars[0]] = ("const", view[1] + start, view[2])
            else:
                self.views[eqn.outvars[0]] = (view[0], view[1] + start)
            return
        raise NotImplementedError(
            f"slice start={start} width={width} strides={strides} only supported "
            "on the parameter/state/const vectors"
        )

    def _squeeze(self, eqn) -> None:
        """Drop a unit axis. On a length-1 vector view this is an index."""
        (a,) = eqn.invars
        view = self.views.get(a)
        if view is not None:
            self.bind(eqn.outvars[0], self._view_element(view, 0))
        else:
            self.bind(eqn.outvars[0], Value(self.resolve(a).name, ""))

    def _convert_element_type(self, eqn) -> None:
        new_dtype = eqn.params["new_dtype"]
        if jnp.dtype(new_dtype) != jnp.float64:
            raise NotImplementedError(f"convert_element_type to {new_dtype}: f64 only")
        self.bind(eqn.outvars[0], self.resolve(eqn.invars[0]))

    def _broadcast_in_dim(self, eqn) -> None:
        """Stretch a value to a larger shape.

        Only two patterns occur for ODE RHS code: a scalar lifted to a tensor
        (``jnp.array([scalar, ...])``), and a rank-preserving no-op. A scalar
        lifted *purely* to feed a ``concatenate`` is recorded and elided — the
        underlying scalar is spliced directly into the concatenation body — so we
        don't emit a dead intermediate vector.
        """
        (in_atom,) = eqn.invars
        in_val = self.resolve(in_atom)
        out_var = eqn.outvars[0]
        out_shape = eqn.params["shape"]
        bcast_dims = eqn.params["broadcast_dimensions"]
        out_rank = len(out_shape)
        out_subs = _index_letters(out_rank)

        if in_val.subs == "" and out_rank >= 1:
            # scalar -> tensor
            self.scalar_wraps[out_var] = in_val.name
            if out_var in self.concat_only:
                self.bind(out_var, Value(in_val.name, ""))
                return
            name = self.fresh("b")
            self.bind(out_var, self.emit(name, out_subs, in_val.name))
            return

        if in_val.rank == out_rank and tuple(bcast_dims) == tuple(range(out_rank)):
            # rank-preserving broadcast: a no-op for our purposes.
            self.bind(out_var, in_val)
            return

        raise NotImplementedError(
            f"broadcast_in_dim shape={out_shape} bcast_dims={bcast_dims} not handled"
        )

    def _concatenate(self, eqn) -> None:
        """Join tensors along axis 0 into a DiffSL tensor literal.

        ``jnp.array([a, b, ...])`` of scalars lowers to per-element broadcasts
        feeding a concatenate. When every input is such an elided scalar we emit
        the compact ``cat_i { a, b }`` form; otherwise we lay each input out at
        its offset, ``cat_i { (0): a, (1:3): b_i }``.
        """
        out_var = eqn.outvars[0]
        out_rank = len(out_var.aval.shape)
        axis = eqn.params["dimension"]
        if out_rank == 0:
            raise NotImplementedError("concatenate of rank-0 inputs is malformed")
        if axis != 0:
            raise NotImplementedError(
                f"concatenate along axis {axis} not supported (only axis 0)"
            )

        out_subs = _index_letters(out_rank)
        out_name = self.fresh("cat")

        all_scalar_wrapped = out_rank == 1 and all(
            iv.aval.shape == (1,) and iv in self.scalar_wraps for iv in eqn.invars
        )
        if all_scalar_wrapped:
            body = ", ".join(self.scalar_wraps[iv] for iv in eqn.invars)
            self.bind(out_var, self.emit(out_name, out_subs, body))
            return

        elements = []
        offset = 0
        for iv in eqn.invars:
            in_shape = iv.aval.shape
            if len(in_shape) != out_rank:
                raise NotImplementedError(
                    f"concatenate input rank {len(in_shape)} != output rank {out_rank}"
                )
            n = in_shape[0]
            ref = self.resolve(iv).ref()
            span = f"({offset})" if n == 1 else f"({offset}:{offset + n})"
            elements.append(f"{span}: {ref}")
            offset += n

        body = ",\n  ".join(elements)
        self.lines.append(f"{out_name}_{out_subs} {{\n  {body},\n}}")
        self.bind(out_var, Value(out_name, out_subs))

    def _call(self, eqn) -> None:
        """Inline a ``pjit`` / ``closed_call`` subjaxpr into the current scope.

        DiffSL has no call construct, so we splice the callee's body in: bind its
        formals to the actual arguments, emit its consts, lower its equations,
        then alias the outer results to the callee's returns.
        """
        inner = eqn.params["jaxpr"]
        if hasattr(inner, "jaxpr"):
            inner_jaxpr = inner.jaxpr
            inner_consts = inner.consts if hasattr(inner, "consts") else inner.literals
        else:
            inner_jaxpr = inner
            inner_consts = []

        for formal, actual in zip(inner_jaxpr.invars, eqn.invars):
            self.bind(formal, self.resolve(actual))
        for cv, cval in zip(inner_jaxpr.constvars, inner_consts):
            self.bind_const(cv, cval)
        self.lower_eqns(inner_jaxpr.eqns)
        for outer_var, ret in zip(eqn.outvars, inner_jaxpr.outvars):
            self.bind(outer_var, self.resolve(ret))

    def _elementwise(self, eqn) -> None:
        """Default handler: scalar/elementwise arithmetic and math functions."""
        prim = eqn.primitive.name
        ref = lambda atom: self.resolve(atom).ref()  # noqa: E731

        if prim in _UNARY_FNS:
            (a,) = eqn.invars
            expr = f"{_UNARY_FNS[prim]}({ref(a)})"
        elif prim in _BINARY_OPS:
            a, b = eqn.invars
            expr = f"{ref(a)} {_BINARY_OPS[prim]} {ref(b)}"
        elif prim == "neg":
            (a,) = eqn.invars
            expr = f"-{ref(a)}"
        elif prim == "pow":
            a, b = eqn.invars
            expr = f"pow({ref(a)}, {ref(b)})"
        elif prim == "integer_pow":
            (a,) = eqn.invars
            expr = f"pow({ref(a)}, {_fmt_float(float(eqn.params['y']))})"
        elif prim == "square":
            r = ref(eqn.invars[0])
            expr = f"{r} * {r}"
        else:
            raise NotImplementedError(f"primitive '{prim}' not supported")

        out_var = eqn.outvars[0]
        out_subs = _index_letters(len(out_var.aval.shape))
        self.bind(out_var, self.emit(self.fresh("v"), out_subs, expr))

    _HANDLERS = {
        "slice": _slice,
        "squeeze": _squeeze,
        "convert_element_type": _convert_element_type,
        "broadcast_in_dim": _broadcast_in_dim,
        "concatenate": _concatenate,
        "pjit": _call,
        "closed_call": _call,
    }


def _find_concat_only_broadcasts(jaxpr) -> set:
    """Vars produced by a scalar ``broadcast_in_dim`` that *only* feed concatenate.

    Such wrappers are pure tracer noise from ``jnp.array([scalar, ...])``: their
    scalar can be spliced straight into the concatenation, so the intermediate
    length-1 vector is never emitted. We descend through ``pjit``/``closed_call``
    and keep a wrapper only if every one of its uses is a concatenate.
    """
    bcast_outs: set = set()
    use_count: dict = {}
    concat_use_count: dict = {}

    def visit(jp):
        for eqn in jp.eqns:
            pn = eqn.primitive.name
            if pn in ("pjit", "closed_call"):
                inner = eqn.params["jaxpr"]
                visit(inner.jaxpr if hasattr(inner, "jaxpr") else inner)
                for iv in eqn.invars:
                    if not isinstance(iv, jex_core.Literal):
                        use_count[iv] = use_count.get(iv, 0) + 1
                continue
            if pn == "broadcast_in_dim":
                (in_var,) = eqn.invars
                in_shape = (
                    () if isinstance(in_var, jex_core.Literal) else in_var.aval.shape
                )
                if in_shape == () and eqn.params["shape"] == (1,):
                    bcast_outs.add(eqn.outvars[0])
            for iv in eqn.invars:
                if isinstance(iv, jex_core.Literal):
                    continue
                use_count[iv] = use_count.get(iv, 0) + 1
                if pn == "concatenate":
                    concat_use_count[iv] = concat_use_count.get(iv, 0) + 1

    visit(jaxpr)

    return {
        v
        for v in bcast_outs
        if use_count.get(v, 0) == concat_use_count.get(v, 0) and use_count.get(v, 0) > 0
    }


def make_diffsl_tuple(
    rhs,
    y0,
    p_example,
    t_example: float = 0.0,
    *,
    param_names: list[str] | None = None,
    state_names: list[str] | None = None,
) -> str:
    """Trace ``rhs`` and emit a DiffSL source string.

    ``rhs(t, y, p)`` must return either a tuple of scalars (one per state
    component) or a single length-``n_state`` vector. ``p`` and ``y`` are indexed
    at the Python level (``p[0]``, ``y[1]``, ...) before any JAX ops.

    The emitted source has four parts: the ``in`` block (parameter defaults), the
    ``u`` block (state initial values, baked in from ``y0``), the body (one tensor
    definition per traced equation), and the ``F`` block (the dudt vector).
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    p_example = jnp.asarray(p_example, dtype=jnp.float64)
    t_example = jnp.float64(t_example)

    n_state = int(y0.shape[0]) if y0.ndim == 1 else 1
    n_param = int(p_example.shape[0]) if p_example.ndim == 1 else 1

    if state_names is None:
        state_names = [f"y{i}" for i in range(n_state)]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_param)]

    closed = jax.make_jaxpr(rhs)(t_example, y0, p_example)
    jaxpr = closed.jaxpr
    consts = closed.literals

    t_var, y_var, p_var = jaxpr.invars
    low = Lowering(
        param_names=param_names,
        state_names=state_names,
        t_var=t_var,
        y_var=y_var,
        p_var=p_var,
    )
    # Bind the three RHS arguments to their DiffSL symbols. t is the reserved
    # ``t``; y and p resolve through slice/squeeze to per-component names, so the
    # vectors themselves only need a placeholder name.
    low.bind(t_var, Value("t"))
    low.bind(y_var, Value("__u_vec", "i" if n_state > 1 else ""))
    low.bind(p_var, Value("__p_vec", "i" if n_param > 1 else ""))
    low.views[y_var] = ("state", 0)
    low.views[p_var] = ("param", 0)
    low.concat_only = _find_concat_only_broadcasts(jaxpr)

    for cv, cval in zip(jaxpr.constvars, consts):
        low.bind_const(cv, cval)
    low.lower_eqns(jaxpr.eqns)

    # ── assemble the four blocks ─────────────────────────────────────────────
    in_subs = "_i" if n_param > 1 else ""
    in_body = ", ".join(
        f"{param_names[i]} = {_fmt_float(float(p_example[i]))}" for i in range(n_param)
    )
    in_line = f"in{in_subs} {{ {in_body} }}"

    if n_state == 1:
        y0_scalar = float(y0[0]) if y0.ndim == 1 else float(y0)
        u_line = f"u {{ {state_names[0]} = {_fmt_float(y0_scalar)} }}"
    else:
        inits = ", ".join(
            f"{state_names[i]} = {_fmt_float(float(y0[i]))}" for i in range(n_state)
        )
        u_line = f"u_i {{ {inits} }}"

    f_line = _emit_f_block(low, jaxpr, n_state)

    body = "\n".join(low.lines)
    return f"{in_line}\n{u_line}\n{body}\n{f_line}\n"


def _emit_f_block(low: Lowering, jaxpr, n_state: int) -> str:
    """Build the ``F`` (dudt) block from the jaxpr's return values.

    ``rhs`` may return either a single length-``n_state`` vector or ``n_state``
    scalars; both are accepted and rendered as ``F`` (scalar state) or ``F_i``.
    """
    outvars = jaxpr.outvars
    single_vec_output = (
        len(outvars) == 1
        and len(outvars[0].aval.shape) == 1
        and outvars[0].aval.shape[0] == n_state
    )
    if not single_vec_output and len(outvars) != n_state:
        raise ValueError(
            f"rhs returned {len(outvars)} outputs of shapes "
            f"{[o.aval.shape for o in outvars]}; expected either "
            f"{n_state} scalars or a single length-{n_state} vector"
        )

    if n_state == 1:
        return f"F {{ {low.resolve(outvars[0]).ref()} }}"

    if single_vec_output:
        val = low.resolve(outvars[0])
        ref = val.name if val.subs == "" else f"{val.name}_i"
        return f"F_i {{ {ref} }}"

    comps = ",\n  ".join(low.resolve(out).name for out in outvars)
    return f"F_i {{\n  {comps},\n}}"
