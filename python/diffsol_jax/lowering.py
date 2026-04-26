from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend import core as jex_core

# skip 'o' (looks like zero), reserve 't'
_INDEX_LETTERS = "ijklmnpqrs"

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

_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _subs_for_rank(rank: int) -> str:
    if rank == 0:
        return ""
    if rank > len(_INDEX_LETTERS):
        raise NotImplementedError(f"rank {rank} exceeds subscript pool")
    return "_" + _INDEX_LETTERS[:rank]


def _fmt_float(x: float) -> str:
    if x == int(x):
        return f"{int(x)}.0"
    return repr(x)


class Env:
    def __init__(self):
        self._by_var: dict = {}
        self._counter = 0

    def fresh_name(self, hint: str = "v") -> str:
        name = f"{hint}{self._counter}"
        self._counter += 1
        return name

    def bind(self, var, name: str, subs: str):
        self._by_var[var] = (name, subs)

    def get(self, atom):
        if isinstance(atom, jex_core.Literal):
            val = atom.val
            if hasattr(val, "shape") and val.shape != ():
                raise NotImplementedError("non-scalar inline literal not supported")
            return _fmt_float(float(val)), ""
        return self._by_var[atom]


def _bind_const(var, val, env: Env, lines: list[str]):
    shape = getattr(val, "shape", ())
    if shape == ():
        name = env.fresh_name("c")
        env.bind(var, name, "")
        lines.append(f"{name} {{ {_fmt_float(float(val))} }}")
    else:
        raise NotImplementedError(
            f"non-scalar constvar (shape={shape}); pass as a parameter instead"
        )


def _find_concat_only_bcasts(jaxpr) -> set:
    """Return the set of vars produced by `broadcast_in_dim` to a length-1
    rank-1 vector that are *only* consumed by `concatenate` (recursively
    descending into pjit / closed_call). Such wrappers can be elided: their
    underlying scalar can be spliced directly into the concatenation body.
    """
    bcast_outs: set = set()
    use_count: dict = {}
    concat_use_count: dict = {}

    def visit(jp):
        for eqn in jp.eqns:
            pn = eqn.primitive.name
            if pn in ("pjit", "closed_call"):
                inner = eqn.params["jaxpr"]
                inner_jp = inner.jaxpr if hasattr(inner, "jaxpr") else inner
                visit(inner_jp)
                # Tally outer-side uses too.
                for iv in eqn.invars:
                    if not isinstance(iv, jex_core.Literal):
                        use_count[iv] = use_count.get(iv, 0) + 1
                continue
            if pn == "broadcast_in_dim":
                (in_var,) = eqn.invars
                out_var = eqn.outvars[0]
                out_shape = eqn.params["shape"]
                in_shape = (
                    in_var.aval.shape
                    if not isinstance(in_var, jex_core.Literal)
                    else ()
                )
                # Only scalar -> length-1 vector wrappers qualify.
                if in_shape == () and out_shape == (1,):
                    bcast_outs.add(out_var)
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


def _emit_eqn(eqn, env: Env, lines: list[str]):
    prim_name = eqn.primitive.name

    if prim_name == "broadcast_in_dim":
        (in_var,) = eqn.invars
        in_name, in_subs = env.get(in_var)
        out_var = eqn.outvars[0]
        out_shape = eqn.params["shape"]
        bcast_dims = eqn.params["broadcast_dimensions"]
        out_rank = len(out_shape)
        out_subs_full = _subs_for_rank(out_rank)[1:] if out_rank else ""
        if in_subs == "" and out_rank >= 1:
            scalar_wraps = getattr(env, "_scalar_wraps", None)
            if scalar_wraps is not None:
                scalar_wraps[out_var] = in_name
            concat_only = getattr(env, "_concat_only_bcasts", None)
            if concat_only is not None and out_var in concat_only:
                env.bind(out_var, in_name, "")
                return
            out_name = env.fresh_name("b")
            env.bind(out_var, out_name, out_subs_full)
            if out_rank == 0:
                lines.append(f"{out_name} {{ {in_name} }}")
            else:
                lines.append(f"{out_name}_{out_subs_full} {{ {in_name} }}")
            return
        if len(in_subs) == out_rank and tuple(bcast_dims) == tuple(range(out_rank)):
            env.bind(out_var, in_name, in_subs)
            return
        raise NotImplementedError(
            f"broadcast_in_dim shape={out_shape} bcast_dims={bcast_dims} not handled"
        )

    if prim_name == "concatenate":
        out_var = eqn.outvars[0]
        out_shape = out_var.aval.shape
        out_rank = len(out_shape)
        axis = eqn.params["dimension"]
        if out_rank == 0:
            raise NotImplementedError("concatenate of rank-0 inputs is malformed")
        if axis != 0:
            raise NotImplementedError(
                f"concatenate along axis {axis} not supported (only axis 0)"
            )

        out_subs_letters = _INDEX_LETTERS[:out_rank]
        out_subs_suffix = f"_{out_subs_letters}"
        out_name = env.fresh_name("cat")

        scalar_wraps = getattr(env, "_scalar_wraps", {})
        all_scalar_wrapped = out_rank == 1 and all(
            len(iv.aval.shape) == 1 and iv.aval.shape[0] == 1 and iv in scalar_wraps
            for iv in eqn.invars
        )
        if all_scalar_wrapped:
            elements = [scalar_wraps[iv] for iv in eqn.invars]
            env.bind(out_var, out_name, out_subs_letters)
            body = ", ".join(elements)
            lines.append(f"{out_name}{out_subs_suffix} {{ {body} }}")
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
            in_name, in_subs = env.get(iv)
            if iv in scalar_wraps and in_subs == "":
                rhs_ref = in_name
            elif in_subs == "":
                rhs_ref = in_name
            else:
                rhs_ref = f"{in_name}_{in_subs}"
            if n == 1:
                elements.append(f"({offset}): {rhs_ref}")
            else:
                elements.append(f"({offset}:{offset + n}): {rhs_ref}")
            offset += n

        env.bind(out_var, out_name, out_subs_letters)
        body = ",\n  ".join(elements)
        lines.append(f"{out_name}{out_subs_suffix} {{\n  {body},\n}}")
        return

    if prim_name == "convert_element_type":
        (in_var,) = eqn.invars
        new_dtype = eqn.params["new_dtype"]
        if jnp.dtype(new_dtype) != jnp.float64:
            raise NotImplementedError(f"convert_element_type to {new_dtype}: f64 only")
        in_name, in_subs = env.get(in_var)
        env.bind(eqn.outvars[0], in_name, in_subs)
        return

    if prim_name in ("pjit", "closed_call"):
        inner = eqn.params["jaxpr"]
        if hasattr(inner, "jaxpr"):
            inner_jaxpr = inner.jaxpr
            inner_consts = inner.consts if hasattr(inner, "consts") else inner.literals
        else:
            inner_jaxpr = inner
            inner_consts = []
        for inner_var, outer_atom in zip(inner_jaxpr.invars, eqn.invars):
            name, subs = env.get(outer_atom)
            env.bind(inner_var, name, subs)
        for cv, cval in zip(inner_jaxpr.constvars, inner_consts):
            _bind_const(cv, cval, env, lines)
        for inner_eqn in inner_jaxpr.eqns:
            _emit_eqn(inner_eqn, env, lines)
        for outer_var, inner_atom in zip(eqn.outvars, inner_jaxpr.outvars):
            if isinstance(inner_atom, jex_core.Literal):
                env.bind(outer_var, _fmt_float(float(inner_atom.val)), "")
            else:
                name, subs = env._by_var[inner_atom]
                env.bind(outer_var, name, subs)
        return

    out_var = eqn.outvars[0]
    out_rank = len(out_var.aval.shape)
    out_subs_letters = _INDEX_LETTERS[:out_rank] if out_rank else ""
    out_subs_suffix = f"_{out_subs_letters}" if out_subs_letters else ""
    out_name = env.fresh_name("v")

    def _ref(atom) -> str:
        name, subs = env.get(atom)
        if isinstance(atom, jex_core.Literal) or subs == "":
            return name
        return f"{name}_{subs}"

    if prim_name in _UNARY_FNS:
        (a,) = eqn.invars
        expr = f"{_UNARY_FNS[prim_name]}({_ref(a)})"
    elif prim_name in _BINARY_OPS:
        a, b = eqn.invars
        expr = f"{_ref(a)} {_BINARY_OPS[prim_name]} {_ref(b)}"
    elif prim_name == "neg":
        (a,) = eqn.invars
        expr = f"-{_ref(a)}"
    elif prim_name == "pow":
        a, b = eqn.invars
        expr = f"pow({_ref(a)}, {_ref(b)})"
    elif prim_name == "integer_pow":
        (a,) = eqn.invars
        expr = f"pow({_ref(a)}, {_fmt_float(float(eqn.params['y']))})"
    elif prim_name == "square":
        (a,) = eqn.invars
        r = _ref(a)
        expr = f"{r} * {r}"
    else:
        raise NotImplementedError(f"primitive '{prim_name}' not supported")

    env.bind(out_var, out_name, out_subs_letters)
    lines.append(f"{out_name}{out_subs_suffix} {{ {expr} }}")


def make_diffsl_tuple(
    rhs,
    y0,
    p_example,
    t_example: float = 0.0,
    *,
    param_names: list[str] | None = None,
    state_names: list[str] | None = None,
) -> str:
    """Trace rhs_tuple and emit a DiffSL source string.

    rhs_tuple(t, y, p) must return a tuple of scalars, one per state component.
    p and y are indexed at the Python level (p[0], y[1], ...) before any JAX ops.
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

    env = Env()
    lines: list[str] = []

    in_body = ", ".join(
        f"{param_names[i]} = {_fmt_float(float(p_example[i]))}" for i in range(n_param)
    )
    in_subs = "_i" if n_param > 1 else ""
    lines.append(f"in{in_subs} {{ {in_body} }}")

    t_var, y_var, p_var = jaxpr.invars
    env.bind(t_var, "t", "")
    env.bind(y_var, "__u_vec", "i" if n_state > 1 else "")
    env.bind(p_var, "__p_vec", "i" if n_param > 1 else "")
    env._param_names_by_index = {i: param_names[i] for i in range(n_param)}
    env._state_names_by_index = {i: state_names[i] for i in range(n_state)}
    env._p_var = p_var
    env._y_var = y_var
    env._scalar_wraps = {}

    env._concat_only_bcasts = _find_concat_only_bcasts(jaxpr)

    for cv, cval in zip(jaxpr.constvars, consts):
        _bind_const(cv, cval, env, lines)

    for eqn in jaxpr.eqns:
        pn = eqn.primitive.name
        if pn == "slice" and eqn.invars[0] in (env._p_var, env._y_var):
            start = eqn.params["start_indices"][0]
            limit = eqn.params["limit_indices"][0]
            if limit - start == 1:
                if eqn.invars[0] is env._p_var:
                    sym = env._param_names_by_index[start]
                else:
                    sym = env._state_names_by_index[start]
                env.bind(eqn.outvars[0], sym, "")
                continue
        if pn == "squeeze":
            (a,) = eqn.invars
            if a is env._p_var and n_param == 1:
                env.bind(eqn.outvars[0], env._param_names_by_index[0], "")
            elif a is env._y_var and n_state == 1:
                env.bind(eqn.outvars[0], env._state_names_by_index[0], "")
            else:
                name, subs = env.get(a)
                env.bind(eqn.outvars[0], name, "")
            continue
        _emit_eqn(eqn, env, lines)

    if n_state == 1:
        y0_scalar = float(y0[0]) if y0.ndim == 1 else float(y0)
        u_line = f"u {{ {state_names[0]} = {_fmt_float(y0_scalar)} }}"
    else:
        inits = ", ".join(
            f"{state_names[i]} = {_fmt_float(float(y0[i]))}" for i in range(n_state)
        )
        u_line = f"u_i {{ {inits} }}"

    n_outvars = len(jaxpr.outvars)
    single_vec_output = (
        n_outvars == 1
        and len(jaxpr.outvars[0].aval.shape) == 1
        and jaxpr.outvars[0].aval.shape[0] == n_state
    )
    if not single_vec_output and n_outvars != n_state:
        raise ValueError(
            f"rhs returned {n_outvars} outputs of shapes "
            f"{[o.aval.shape for o in jaxpr.outvars]}; expected either "
            f"{n_state} scalars or a single length-{n_state} vector"
        )
    if n_state == 1:
        if single_vec_output:
            name, subs = env.get(jaxpr.outvars[0])
            ref = name if subs == "" else f"{name}_{subs}"
            f_line = f"F {{ {ref} }}"
        else:
            name, _ = env.get(jaxpr.outvars[0])
            f_line = f"F {{ {name} }}"
    else:
        if single_vec_output:
            name, subs = env.get(jaxpr.outvars[0])
            if subs == "":
                ref = name
            else:
                ref = f"{name}_i"
            f_line = f"F_i {{ {ref} }}"
        else:
            comps = [env.get(out)[0] for out in jaxpr.outvars]
            f_line = "F_i {\n  " + ",\n  ".join(comps) + ",\n}"

    header = lines[0]
    body = "\n".join(lines[1:])
    return f"{header}\n{u_line}\n{body}\n{f_line}\n"
