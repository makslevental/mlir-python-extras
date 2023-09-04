from textwrap import dedent

from mlir.dialects._nvgpu_enum_gen import (
    TensorMapSwizzleKind,
    TensorMapL2PromoKind,
    TensorMapOOBKind,
    TensorMapInterleaveKind,
)
from mlir.ir import Attribute, Type


def tensormap_descriptor(
    tensor,
    swizzle: TensorMapSwizzleKind = TensorMapSwizzleKind.SWIZZLE_NONE,
    l2promo: TensorMapL2PromoKind = TensorMapL2PromoKind.L2PROMO_NONE,
    oob: TensorMapOOBKind = TensorMapOOBKind.OOB_NAN,
    interleave: TensorMapInterleaveKind = TensorMapInterleaveKind.INTERLEAVE_NONE,
    context=None,
):
    return Type.parse(
        dedent(
            f"""\
                !nvgpu.tensormap.descriptor<tensor = {tensor}, 
                                            swizzle = {swizzle}, 
                                            l2promo = {l2promo}, 
                                            oob = {oob}, 
                                            interleave = {interleave}>
            """
        ),
        context=context,
    )
