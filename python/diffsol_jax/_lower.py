import functools

import jax

from ._equations import FULL_ROLES, SCALAR_ROLES, CallbackId, Equations

FWD_SENS_MULTI_P = (CallbackId.RHS_JAC, CallbackId.OUT_JAC, CallbackId.MASS)
VJP_BKWD_MULTI_P = (
    CallbackId.RHS_ADJOINT,
    CallbackId.RHS_SENS_ADJOINT,
    CallbackId.OUT_ADJOINT,
    CallbackId.OUT_SENS_ADJOINT,
    CallbackId.INIT_SENS_ADJOINT,
    CallbackId.MASS_ADJOINT,
)


def batched(eq: Equations, cb_id: CallbackId):
    cb = eq.callbacks()[cb_id]
    shape = {
        "x": lambda B, P: (B, eq.nstates),
        "p": lambda B, P: (B, eq.nparams),
        "t": lambda B, P: (),
        "beta": lambda B, P: (),
        "v": lambda B, P: (B, P, cb.nv),
        "y": lambda B, P: (B, P, cb.ny),
    }

    fn = cb.fn
    if any(role in FULL_ROLES for role in cb.args):
        fn = jax.vmap(fn, tuple(0 if r in FULL_ROLES else None for r in cb.args))
    all_lanes = jax.vmap(fn, tuple(None if r in SCALAR_ROLES else 0 for r in cb.args))

    def shapes(B, P):
        return tuple(
            jax.ShapeDtypeStruct(shape[role](B, P), eq.dtype) for role in cb.args
        )

    return all_lanes, shapes


def lower(eq: Equations, cb_id: CallbackId, B: int, P: int) -> bytes:
    """The StableHLO of callback `cb_id` for `B` lanes and `P` lanes on the `P` axis."""
    from jax._src.lib.mlir.dialects import hlo

    all_lanes, shapes = batched(eq, cb_id)
    lowered = jax.jit(all_lanes, keep_unused=True).lower(*shapes(B, P))
    module = lowered.compiler_ir("stablehlo")
    return hlo.serialize_portable_artifact_str(str(module), hlo.get_minimum_version())


@functools.cache
def compile_options() -> bytes:
    from jax._src.lib import xla_client

    return xla_client.CompileOptions().SerializeAsString()
