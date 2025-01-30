# noinspection PyUnresolvedReferences
from .....ir import Type, Value

from typing import NewType, TypeVar, Generic, Literal
from . import call_intrinsic

ValueRef = Value

any = NewType("any", ValueRef)
anyfloat = NewType("anyfloat", ValueRef)
anyint = NewType("anyint", ValueRef)
anyptr = NewType("anyptr", ValueRef)
anyvector = NewType("anyvector", ValueRef)
bf16 = NewType("bf16", ValueRef)
f64 = NewType("f64", ValueRef)
f128 = NewType("f128", ValueRef)
f80 = NewType("f80", ValueRef)
f32 = NewType("f32", ValueRef)
half = NewType("half", ValueRef)
i1 = NewType("i1", ValueRef)
i128 = NewType("i128", ValueRef)
i16 = NewType("i16", ValueRef)
i32 = NewType("i32", ValueRef)
i64 = NewType("i64", ValueRef)
i8 = NewType("i8", ValueRef)
ppcf128 = NewType("ppcf128", ValueRef)
pointer = NewType("pointer", ValueRef)
void = NewType("void", ValueRef)

v1i1 = NewType("v1i1", ValueRef)
v2i1 = NewType("v2i1", ValueRef)
v3i1 = NewType("v3i1", ValueRef)
v4i1 = NewType("v4i1", ValueRef)
v8i1 = NewType("v8i1", ValueRef)
v16i1 = NewType("v16i1", ValueRef)
v32i1 = NewType("v32i1", ValueRef)
v64i1 = NewType("v64i1", ValueRef)
v128i1 = NewType("v128i1", ValueRef)
v256i1 = NewType("v256i1", ValueRef)
v512i1 = NewType("v512i1", ValueRef)
v1024i1 = NewType("v1024i1", ValueRef)
v2048i1 = NewType("v2048i1", ValueRef)

v128i2 = NewType("v128i2", ValueRef)
v256i2 = NewType("v256i2", ValueRef)

v64i4 = NewType("v64i4", ValueRef)
v128i4 = NewType("v128i4", ValueRef)

v1i8 = NewType("v1i8", ValueRef)
v2i8 = NewType("v2i8", ValueRef)
v3i8 = NewType("v3i8", ValueRef)
v4i8 = NewType("v4i8", ValueRef)
v8i8 = NewType("v8i8", ValueRef)
v16i8 = NewType("v16i8", ValueRef)
v32i8 = NewType("v32i8", ValueRef)
v64i8 = NewType("v64i8", ValueRef)
v128i8 = NewType("v128i8", ValueRef)
v256i8 = NewType("v256i8", ValueRef)
v512i8 = NewType("v512i8", ValueRef)
v1024i8 = NewType("v1024i8", ValueRef)

v1i16 = NewType("v1i16", ValueRef)
v2i16 = NewType("v2i16", ValueRef)
v3i16 = NewType("v3i16", ValueRef)
v4i16 = NewType("v4i16", ValueRef)
v8i16 = NewType("v8i16", ValueRef)
v16i16 = NewType("v16i16", ValueRef)
v32i16 = NewType("v32i16", ValueRef)
v64i16 = NewType("v64i16", ValueRef)
v128i16 = NewType("v128i16", ValueRef)
v256i16 = NewType("v256i16", ValueRef)
v512i16 = NewType("v512i16", ValueRef)

v1i32 = NewType("v1i32", ValueRef)
v2i32 = NewType("v2i32", ValueRef)
v3i32 = NewType("v3i32", ValueRef)
v4i32 = NewType("v4i32", ValueRef)
v5i32 = NewType("v5i32", ValueRef)
v6i32 = NewType("v6i32", ValueRef)
v7i32 = NewType("v7i32", ValueRef)
v8i32 = NewType("v8i32", ValueRef)
v9i32 = NewType("v9i32", ValueRef)
v10i32 = NewType("v10i32", ValueRef)
v11i32 = NewType("v11i32", ValueRef)
v12i32 = NewType("v12i32", ValueRef)
v16i32 = NewType("v16i32", ValueRef)
v32i32 = NewType("v32i32", ValueRef)
v64i32 = NewType("v64i32", ValueRef)
v128i32 = NewType("v128i32", ValueRef)
v256i32 = NewType("v256i32", ValueRef)
v512i32 = NewType("v512i32", ValueRef)
v1024i32 = NewType("v1024i32", ValueRef)
v2048i32 = NewType("v2048i32", ValueRef)

v1i64 = NewType("v1i64", ValueRef)
v2i64 = NewType("v2i64", ValueRef)
v3i64 = NewType("v3i64", ValueRef)
v4i64 = NewType("v4i64", ValueRef)
v8i64 = NewType("v8i64", ValueRef)
v16i64 = NewType("v16i64", ValueRef)
v32i64 = NewType("v32i64", ValueRef)
v64i64 = NewType("v64i64", ValueRef)
v128i64 = NewType("v128i64", ValueRef)
v256i64 = NewType("v256i64", ValueRef)

v1i128 = NewType("v1i128", ValueRef)

v1f16 = NewType("v1f16", ValueRef)
v2f16 = NewType("v2f16", ValueRef)
v3f16 = NewType("v3f16", ValueRef)
v4f16 = NewType("v4f16", ValueRef)
v8f16 = NewType("v8f16", ValueRef)
v16f16 = NewType("v16f16", ValueRef)
v32f16 = NewType("v32f16", ValueRef)
v64f16 = NewType("v64f16", ValueRef)
v128f16 = NewType("v128f16", ValueRef)
v256f16 = NewType("v256f16", ValueRef)
v512f16 = NewType("v512f16", ValueRef)

v1bf16 = NewType("v1bf16", ValueRef)
v2bf16 = NewType("v2bf16", ValueRef)
v3bf16 = NewType("v3bf16", ValueRef)
v4bf16 = NewType("v4bf16", ValueRef)
v8bf16 = NewType("v8bf16", ValueRef)
v16bf16 = NewType("v16bf16", ValueRef)
v32bf16 = NewType("v32bf16", ValueRef)
v64bf16 = NewType("v64bf16", ValueRef)
v128bf16 = NewType("v128bf16", ValueRef)

v1f32 = NewType("v1f32", ValueRef)
v2f32 = NewType("v2f32", ValueRef)
v3f32 = NewType("v3f32", ValueRef)
v4f32 = NewType("v4f32", ValueRef)
v5f32 = NewType("v5f32", ValueRef)
v6f32 = NewType("v6f32", ValueRef)
v7f32 = NewType("v7f32", ValueRef)
v8f32 = NewType("v8f32", ValueRef)
v9f32 = NewType("v9f32", ValueRef)
v10f32 = NewType("v10f32", ValueRef)
v11f32 = NewType("v11f32", ValueRef)
v12f32 = NewType("v12f32", ValueRef)
v16f32 = NewType("v16f32", ValueRef)
v32f32 = NewType("v32f32", ValueRef)
v64f32 = NewType("v64f32", ValueRef)
v128f32 = NewType("v128f32", ValueRef)
v256f32 = NewType("v256f32", ValueRef)
v512f32 = NewType("v512f32", ValueRef)
v1024f32 = NewType("v1024f32", ValueRef)
v2048f32 = NewType("v2048f32", ValueRef)

v1f64 = NewType("v1f64", ValueRef)
v2f64 = NewType("v2f64", ValueRef)
v3f64 = NewType("v3f64", ValueRef)
v4f64 = NewType("v4f64", ValueRef)
v8f64 = NewType("v8f64", ValueRef)
v16f64 = NewType("v16f64", ValueRef)
v32f64 = NewType("v32f64", ValueRef)
v64f64 = NewType("v64f64", ValueRef)
v128f64 = NewType("v128f64", ValueRef)
v256f64 = NewType("v256f64", ValueRef)

vararg = NewType("vararg", ValueRef)
metadata = NewType("metadata", ValueRef)

_T = TypeVar("_T")


class LLVMQualPointerType(Generic[_T]):
    pass


local_ptr = LLVMQualPointerType[Literal[3]]
global_ptr = LLVMQualPointerType[Literal[1]]
AMDGPUBufferRsrcTy = LLVMQualPointerType[Literal[8]]


class LLVMMatchType(Generic[_T]):
    pass



def addrspacecast_nonnull(a: anyptr, return_type=None):
    return call_intrinsic(a, intr_id=2051, intr_name="llvm.amdgcn.addrspacecast.nonnull", is_overloaded=True, return_type=return_type)


def alignbyte(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2052, intr_name="llvm.amdgcn.alignbyte", is_overloaded=False, return_type=return_type)


def ashr_pk_i8_i32(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2053, intr_name="llvm.amdgcn.ashr.pk.i8.i32", is_overloaded=False, return_type=return_type)


def ashr_pk_u8_i32(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2054, intr_name="llvm.amdgcn.ashr.pk.u8.i32", is_overloaded=False, return_type=return_type)


def atomic_cond_sub_u32(a: anyptr, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2055, intr_name="llvm.amdgcn.atomic.cond.sub.u32", is_overloaded=True, return_type=return_type)


def ballot(a: i1, return_type=None):
    return call_intrinsic(a, intr_id=2056, intr_name="llvm.amdgcn.ballot", is_overloaded=True, return_type=return_type)


def bitop3(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: LLVMMatchType[Literal[0]], d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2057, intr_name="llvm.amdgcn.bitop3", is_overloaded=True, return_type=return_type)


def buffer_wbinvl1(return_type=None):
    call_intrinsic(intr_id=2058, intr_name="llvm.amdgcn.buffer.wbinvl1", is_overloaded=False, return_type=return_type)


def buffer_wbinvl1_sc(return_type=None):
    call_intrinsic(intr_id=2059, intr_name="llvm.amdgcn.buffer.wbinvl1.sc", is_overloaded=False, return_type=return_type)


def buffer_wbinvl1_vol(return_type=None):
    call_intrinsic(intr_id=2060, intr_name="llvm.amdgcn.buffer.wbinvl1.vol", is_overloaded=False, return_type=return_type)


def class_(a: anyfloat, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2061, intr_name="llvm.amdgcn.class", is_overloaded=True, return_type=return_type)


def cos(a: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, intr_id=2062, intr_name="llvm.amdgcn.cos", is_overloaded=True, return_type=return_type)


def cs_chain(a: anyptr, b: anyint, c: any, d: any, e: i32, f: vararg, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=2063, intr_name="llvm.amdgcn.cs.chain", is_overloaded=True, return_type=return_type)


def cubeid(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2064, intr_name="llvm.amdgcn.cubeid", is_overloaded=False, return_type=return_type)


def cubema(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2065, intr_name="llvm.amdgcn.cubema", is_overloaded=False, return_type=return_type)


def cubesc(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2066, intr_name="llvm.amdgcn.cubesc", is_overloaded=False, return_type=return_type)


def cubetc(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2067, intr_name="llvm.amdgcn.cubetc", is_overloaded=False, return_type=return_type)


def cvt_f32_bf8(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2068, intr_name="llvm.amdgcn.cvt.f32.bf8", is_overloaded=False, return_type=return_type)


def cvt_f32_fp8(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2069, intr_name="llvm.amdgcn.cvt.f32.fp8", is_overloaded=False, return_type=return_type)


def cvt_pk_bf8_f32(a: float, b: float, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2070, intr_name="llvm.amdgcn.cvt.pk.bf8.f32", is_overloaded=False, return_type=return_type)


def cvt_pk_f32_bf8(a: i32, b: i1, return_type=None):
    return call_intrinsic(a, b, intr_id=2071, intr_name="llvm.amdgcn.cvt.pk.f32.bf8", is_overloaded=False, return_type=return_type)


def cvt_pk_f32_fp8(a: i32, b: i1, return_type=None):
    return call_intrinsic(a, b, intr_id=2072, intr_name="llvm.amdgcn.cvt.pk.f32.fp8", is_overloaded=False, return_type=return_type)


def cvt_pk_fp8_f32(a: float, b: float, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2073, intr_name="llvm.amdgcn.cvt.pk.fp8.f32", is_overloaded=False, return_type=return_type)


def cvt_pk_i16(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2074, intr_name="llvm.amdgcn.cvt.pk.i16", is_overloaded=False, return_type=return_type)


def cvt_pk_u16(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2075, intr_name="llvm.amdgcn.cvt.pk.u16", is_overloaded=False, return_type=return_type)


def cvt_pk_u8_f32(a: float, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2076, intr_name="llvm.amdgcn.cvt.pk.u8.f32", is_overloaded=False, return_type=return_type)


def cvt_pknorm_i16(a: float, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2077, intr_name="llvm.amdgcn.cvt.pknorm.i16", is_overloaded=False, return_type=return_type)


def cvt_pknorm_u16(a: float, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2078, intr_name="llvm.amdgcn.cvt.pknorm.u16", is_overloaded=False, return_type=return_type)


def cvt_pkrtz(a: float, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2079, intr_name="llvm.amdgcn.cvt.pkrtz", is_overloaded=False, return_type=return_type)


def cvt_scalef32_2xpk16_bf6_f32(a: v16f32, b: v16f32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2080, intr_name="llvm.amdgcn.cvt.scalef32.2xpk16.bf6.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_2xpk16_fp6_f32(a: v16f32, b: v16f32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2081, intr_name="llvm.amdgcn.cvt.scalef32.2xpk16.fp6.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_f16_bf8(a: v2f16, b: i32, c: float, d: i32, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2082, intr_name="llvm.amdgcn.cvt.scalef32.f16.bf8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_f16_fp8(a: v2f16, b: i32, c: float, d: i32, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2083, intr_name="llvm.amdgcn.cvt.scalef32.f16.fp8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_f32_bf8(a: i32, b: float, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2084, intr_name="llvm.amdgcn.cvt.scalef32.f32.bf8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_f32_fp8(a: i32, b: float, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2085, intr_name="llvm.amdgcn.cvt.scalef32.f32.fp8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_bf16_bf6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2104, intr_name="llvm.amdgcn.cvt.scalef32.pk32.bf16.bf6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_bf16_fp6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2105, intr_name="llvm.amdgcn.cvt.scalef32.pk32.bf16.fp6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_bf6_bf16(a: v32bf16, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2106, intr_name="llvm.amdgcn.cvt.scalef32.pk32.bf6.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_bf6_f16(a: v32f16, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2107, intr_name="llvm.amdgcn.cvt.scalef32.pk32.bf6.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_f16_bf6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2108, intr_name="llvm.amdgcn.cvt.scalef32.pk32.f16.bf6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_f16_fp6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2109, intr_name="llvm.amdgcn.cvt.scalef32.pk32.f16.fp6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_f32_bf6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2110, intr_name="llvm.amdgcn.cvt.scalef32.pk32.f32.bf6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_f32_fp6(a: v6i32, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2111, intr_name="llvm.amdgcn.cvt.scalef32.pk32.f32.fp6", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_fp6_bf16(a: v32bf16, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2112, intr_name="llvm.amdgcn.cvt.scalef32.pk32.fp6.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk32_fp6_f16(a: v32f16, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2113, intr_name="llvm.amdgcn.cvt.scalef32.pk32.fp6.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf16_bf8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2086, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf16.bf8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf16_fp4(a: i32, b: float, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2087, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf16.fp4", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf16_fp8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2088, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf16.fp8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf8_bf16(a: v2i16, b: v2bf16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2089, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf8.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf8_f16(a: v2i16, b: v2f16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2090, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf8.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_bf8_f32(a: v2i16, b: float, c: float, d: float, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2091, intr_name="llvm.amdgcn.cvt.scalef32.pk.bf8.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f16_bf8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2092, intr_name="llvm.amdgcn.cvt.scalef32.pk.f16.bf8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f16_fp4(a: i32, b: float, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2093, intr_name="llvm.amdgcn.cvt.scalef32.pk.f16.fp4", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f16_fp8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2094, intr_name="llvm.amdgcn.cvt.scalef32.pk.f16.fp8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f32_bf8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2095, intr_name="llvm.amdgcn.cvt.scalef32.pk.f32.bf8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f32_fp4(a: i32, b: float, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2096, intr_name="llvm.amdgcn.cvt.scalef32.pk.f32.fp4", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_f32_fp8(a: i32, b: float, c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2097, intr_name="llvm.amdgcn.cvt.scalef32.pk.f32.fp8", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp4_bf16(a: i32, b: v2bf16, c: float, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2098, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp4.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp4_f16(a: i32, b: v2f16, c: float, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2099, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp4.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp4_f32(a: i32, b: float, c: float, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2100, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp4.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp8_bf16(a: v2i16, b: v2bf16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2101, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp8.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp8_f16(a: v2i16, b: v2f16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2102, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp8.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_pk_fp8_f32(a: v2i16, b: float, c: float, d: float, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2103, intr_name="llvm.amdgcn.cvt.scalef32.pk.fp8.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_bf8_bf16(a: i32, b: bf16, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2114, intr_name="llvm.amdgcn.cvt.scalef32.sr.bf8.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_bf8_f16(a: i32, b: half, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2115, intr_name="llvm.amdgcn.cvt.scalef32.sr.bf8.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_bf8_f32(a: i32, b: float, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2116, intr_name="llvm.amdgcn.cvt.scalef32.sr.bf8.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_fp8_bf16(a: i32, b: bf16, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2117, intr_name="llvm.amdgcn.cvt.scalef32.sr.fp8.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_fp8_f16(a: i32, b: half, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2118, intr_name="llvm.amdgcn.cvt.scalef32.sr.fp8.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_fp8_f32(a: i32, b: float, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2119, intr_name="llvm.amdgcn.cvt.scalef32.sr.fp8.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_bf6_bf16(a: v32bf16, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2123, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_bf6_f16(a: v32f16, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2124, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_bf6_f32(a: v32f32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2125, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_fp6_bf16(a: v32bf16, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2126, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_fp6_f16(a: v32f16, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2127, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk32_fp6_f32(a: v32f32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2128, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.f32", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk_fp4_bf16(a: i32, b: v2bf16, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2120, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk.fp4.bf16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk_fp4_f16(a: i32, b: v2f16, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2121, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16", is_overloaded=False, return_type=return_type)


def cvt_scalef32_sr_pk_fp4_f32(a: i32, b: v2f32, c: i32, d: float, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2122, intr_name="llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f32", is_overloaded=False, return_type=return_type)


def cvt_sr_bf16_f32(a: v2bf16, b: float, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2129, intr_name="llvm.amdgcn.cvt.sr.bf16.f32", is_overloaded=False, return_type=return_type)


def cvt_sr_bf8_f32(a: float, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2130, intr_name="llvm.amdgcn.cvt.sr.bf8.f32", is_overloaded=False, return_type=return_type)


def cvt_sr_f16_f32(a: v2f16, b: float, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2131, intr_name="llvm.amdgcn.cvt.sr.f16.f32", is_overloaded=False, return_type=return_type)


def cvt_sr_fp8_f32(a: float, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2132, intr_name="llvm.amdgcn.cvt.sr.fp8.f32", is_overloaded=False, return_type=return_type)


def dispatch_id(return_type=None):
    return call_intrinsic(intr_id=2133, intr_name="llvm.amdgcn.dispatch.id", is_overloaded=False, return_type=return_type)


def dispatch_ptr(return_type=None):
    return call_intrinsic(intr_id=2134, intr_name="llvm.amdgcn.dispatch.ptr", is_overloaded=False, return_type=return_type)


def div_fixup(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, b, c, intr_id=2135, intr_name="llvm.amdgcn.div.fixup", is_overloaded=True, return_type=return_type)


def div_fmas(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: LLVMMatchType[Literal[0]], d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2136, intr_name="llvm.amdgcn.div.fmas", is_overloaded=True, return_type=return_type)


def div_scale(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: i1, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2137, intr_name="llvm.amdgcn.div.scale", is_overloaded=True, return_type=return_type)


def dot4_f32_bf8_bf8(a: i32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2138, intr_name="llvm.amdgcn.dot4.f32.bf8.bf8", is_overloaded=False, return_type=return_type)


def dot4_f32_bf8_fp8(a: i32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2139, intr_name="llvm.amdgcn.dot4.f32.bf8.fp8", is_overloaded=False, return_type=return_type)


def dot4_f32_fp8_bf8(a: i32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2140, intr_name="llvm.amdgcn.dot4.f32.fp8.bf8", is_overloaded=False, return_type=return_type)


def dot4_f32_fp8_fp8(a: i32, b: i32, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2141, intr_name="llvm.amdgcn.dot4.f32.fp8.fp8", is_overloaded=False, return_type=return_type)


def ds_add_gs_reg_rtn(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2142, intr_name="llvm.amdgcn.ds.add.gs.reg.rtn", is_overloaded=True, return_type=return_type)


def ds_append(a: anyptr, b: i1, return_type=None):
    return call_intrinsic(a, b, intr_id=2143, intr_name="llvm.amdgcn.ds.append", is_overloaded=True, return_type=return_type)


def ds_bpermute(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2144, intr_name="llvm.amdgcn.ds.bpermute", is_overloaded=False, return_type=return_type)


def ds_bvh_stack_rtn(a: i32, b: i32, c: v4i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2145, intr_name="llvm.amdgcn.ds.bvh.stack.rtn", is_overloaded=False, return_type=return_type)


def ds_consume(a: anyptr, b: i1, return_type=None):
    return call_intrinsic(a, b, intr_id=2146, intr_name="llvm.amdgcn.ds.consume", is_overloaded=True, return_type=return_type)


def ds_gws_barrier(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=2147, intr_name="llvm.amdgcn.ds.gws.barrier", is_overloaded=False, return_type=return_type)


def ds_gws_init(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=2148, intr_name="llvm.amdgcn.ds.gws.init", is_overloaded=False, return_type=return_type)


def ds_gws_sema_br(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=2149, intr_name="llvm.amdgcn.ds.gws.sema.br", is_overloaded=False, return_type=return_type)


def ds_gws_sema_p(a: i32, return_type=None):
    call_intrinsic(a, intr_id=2150, intr_name="llvm.amdgcn.ds.gws.sema.p", is_overloaded=False, return_type=return_type)


def ds_gws_sema_release_all(a: i32, return_type=None):
    call_intrinsic(a, intr_id=2151, intr_name="llvm.amdgcn.ds.gws.sema.release.all", is_overloaded=False, return_type=return_type)


def ds_gws_sema_v(a: i32, return_type=None):
    call_intrinsic(a, intr_id=2152, intr_name="llvm.amdgcn.ds.gws.sema.v", is_overloaded=False, return_type=return_type)


def ds_ordered_add(a: LLVMQualPointerType[Literal[2]], b: i32, c: i32, d: i32, e: i1, f: i32, g: i1, h: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2153, intr_name="llvm.amdgcn.ds.ordered.add", is_overloaded=False, return_type=return_type)


def ds_ordered_swap(a: LLVMQualPointerType[Literal[2]], b: i32, c: i32, d: i32, e: i1, f: i32, g: i1, h: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2154, intr_name="llvm.amdgcn.ds.ordered.swap", is_overloaded=False, return_type=return_type)


def ds_permute(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2155, intr_name="llvm.amdgcn.ds.permute", is_overloaded=False, return_type=return_type)


def ds_read_tr16_b64(a: local_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2156, intr_name="llvm.amdgcn.ds.read.tr16.b64", is_overloaded=True, return_type=return_type)


def ds_read_tr4_b64(a: local_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2157, intr_name="llvm.amdgcn.ds.read.tr4.b64", is_overloaded=True, return_type=return_type)


def ds_read_tr6_b96(a: local_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2158, intr_name="llvm.amdgcn.ds.read.tr6.b96", is_overloaded=True, return_type=return_type)


def ds_read_tr8_b64(a: local_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2159, intr_name="llvm.amdgcn.ds.read.tr8.b64", is_overloaded=True, return_type=return_type)


def ds_sub_gs_reg_rtn(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2160, intr_name="llvm.amdgcn.ds.sub.gs.reg.rtn", is_overloaded=True, return_type=return_type)


def ds_swizzle(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2161, intr_name="llvm.amdgcn.ds.swizzle", is_overloaded=False, return_type=return_type)


def else_(a: anyint, return_type=None):
    return call_intrinsic(a, intr_id=2162, intr_name="llvm.amdgcn.else", is_overloaded=True, return_type=return_type)


def end_cf(a: anyint, return_type=None):
    call_intrinsic(a, intr_id=2163, intr_name="llvm.amdgcn.end.cf", is_overloaded=True, return_type=return_type)


def endpgm(return_type=None):
    call_intrinsic(intr_id=2164, intr_name="llvm.amdgcn.endpgm", is_overloaded=False, return_type=return_type)


def exp(a: i32, b: i32, c: any, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: i1, h: i1, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2165, intr_name="llvm.amdgcn.exp", is_overloaded=True, return_type=return_type)


def exp2(a: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, intr_id=2168, intr_name="llvm.amdgcn.exp2", is_overloaded=True, return_type=return_type)


def exp_compr(a: i32, b: i32, c: anyvector, d: LLVMMatchType[Literal[0]], e: i1, f: i1, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=2166, intr_name="llvm.amdgcn.exp.compr", is_overloaded=True, return_type=return_type)


def exp_row(a: i32, b: i32, c: any, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: i1, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2167, intr_name="llvm.amdgcn.exp.row", is_overloaded=True, return_type=return_type)


def fcmp(a: anyfloat, b: LLVMMatchType[Literal[1]], c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2169, intr_name="llvm.amdgcn.fcmp", is_overloaded=True, return_type=return_type)


def fdiv_fast(a: float, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2170, intr_name="llvm.amdgcn.fdiv.fast", is_overloaded=False, return_type=return_type)


def fdot2(a: v2f16, b: v2f16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2171, intr_name="llvm.amdgcn.fdot2", is_overloaded=False, return_type=return_type)


def fdot2_bf16_bf16(a: v2bf16, b: v2bf16, c: bf16, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2172, intr_name="llvm.amdgcn.fdot2.bf16.bf16", is_overloaded=False, return_type=return_type)


def fdot2_f16_f16(a: v2f16, b: v2f16, c: half, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2173, intr_name="llvm.amdgcn.fdot2.f16.f16", is_overloaded=False, return_type=return_type)


def fdot2_f32_bf16(a: v2bf16, b: v2bf16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2174, intr_name="llvm.amdgcn.fdot2.f32.bf16", is_overloaded=False, return_type=return_type)


def fdot2c_f32_bf16(a: v2bf16, b: v2bf16, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2175, intr_name="llvm.amdgcn.fdot2c.f32.bf16", is_overloaded=False, return_type=return_type)


def flat_atomic_fmax_num(a: anyptr, b: anyfloat, return_type=None):
    return call_intrinsic(a, b, intr_id=2176, intr_name="llvm.amdgcn.flat.atomic.fmax.num", is_overloaded=True, return_type=return_type)


def flat_atomic_fmin_num(a: anyptr, b: anyfloat, return_type=None):
    return call_intrinsic(a, b, intr_id=2177, intr_name="llvm.amdgcn.flat.atomic.fmin.num", is_overloaded=True, return_type=return_type)


def fma_legacy(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2178, intr_name="llvm.amdgcn.fma.legacy", is_overloaded=False, return_type=return_type)


def fmad_ftz(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, b, c, intr_id=2179, intr_name="llvm.amdgcn.fmad.ftz", is_overloaded=True, return_type=return_type)


def fmed3(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, b, c, intr_id=2180, intr_name="llvm.amdgcn.fmed3", is_overloaded=True, return_type=return_type)


def fmul_legacy(a: float, b: float, return_type=None):
    return call_intrinsic(a, b, intr_id=2181, intr_name="llvm.amdgcn.fmul.legacy", is_overloaded=False, return_type=return_type)


def fract(a: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, intr_id=2182, intr_name="llvm.amdgcn.fract", is_overloaded=True, return_type=return_type)


def frexp_exp(a: anyfloat, return_type=None):
    return call_intrinsic(a, intr_id=2183, intr_name="llvm.amdgcn.frexp.exp", is_overloaded=True, return_type=return_type)


def frexp_mant(a: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, intr_id=2184, intr_name="llvm.amdgcn.frexp.mant", is_overloaded=True, return_type=return_type)


def global_atomic_csub(a: anyptr, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2185, intr_name="llvm.amdgcn.global.atomic.csub", is_overloaded=True, return_type=return_type)


def global_atomic_fmax_num(a: anyptr, b: anyfloat, return_type=None):
    return call_intrinsic(a, b, intr_id=2186, intr_name="llvm.amdgcn.global.atomic.fmax.num", is_overloaded=True, return_type=return_type)


def global_atomic_fmin_num(a: anyptr, b: anyfloat, return_type=None):
    return call_intrinsic(a, b, intr_id=2187, intr_name="llvm.amdgcn.global.atomic.fmin.num", is_overloaded=True, return_type=return_type)


def global_atomic_ordered_add_b64(a: global_ptr, b: i64, return_type=None):
    return call_intrinsic(a, b, intr_id=2188, intr_name="llvm.amdgcn.global.atomic.ordered.add.b64", is_overloaded=False, return_type=return_type)


def global_load_lds(a: LLVMQualPointerType[Literal[1]], b: LLVMQualPointerType[Literal[3]], c: i32, d: i32, e: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, intr_id=2189, intr_name="llvm.amdgcn.global.load.lds", is_overloaded=False, return_type=return_type)


def global_load_tr_b128(a: global_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2190, intr_name="llvm.amdgcn.global.load.tr.b128", is_overloaded=True, return_type=return_type)


def global_load_tr_b64(a: global_ptr, return_type=None):
    return call_intrinsic(a, intr_id=2191, intr_name="llvm.amdgcn.global.load.tr.b64", is_overloaded=True, return_type=return_type)


def groupstaticsize(return_type=None):
    return call_intrinsic(intr_id=2192, intr_name="llvm.amdgcn.groupstaticsize", is_overloaded=False, return_type=return_type)


def icmp(a: anyint, b: LLVMMatchType[Literal[1]], c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2193, intr_name="llvm.amdgcn.icmp", is_overloaded=True, return_type=return_type)


def if_(a: i1, return_type=None):
    return call_intrinsic(a, intr_id=2194, intr_name="llvm.amdgcn.if", is_overloaded=True, return_type=return_type)


def if_break(a: i1, b: LLVMMatchType[Literal[0]], return_type=None):
    return call_intrinsic(a, b, intr_id=2195, intr_name="llvm.amdgcn.if.break", is_overloaded=True, return_type=return_type)


def iglp_opt(a: i32, return_type=None):
    call_intrinsic(a, intr_id=2196, intr_name="llvm.amdgcn.iglp.opt", is_overloaded=False, return_type=return_type)


def image_atomic_add_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2197, intr_name="llvm.amdgcn.image.atomic.add.1d", is_overloaded=True, return_type=return_type)


def image_atomic_add_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2198, intr_name="llvm.amdgcn.image.atomic.add.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_add_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2199, intr_name="llvm.amdgcn.image.atomic.add.2d", is_overloaded=True, return_type=return_type)


def image_atomic_add_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2200, intr_name="llvm.amdgcn.image.atomic.add.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_add_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2201, intr_name="llvm.amdgcn.image.atomic.add.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_add_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2202, intr_name="llvm.amdgcn.image.atomic.add.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_add_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2203, intr_name="llvm.amdgcn.image.atomic.add.3d", is_overloaded=True, return_type=return_type)


def image_atomic_add_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2204, intr_name="llvm.amdgcn.image.atomic.add.cube", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2205, intr_name="llvm.amdgcn.image.atomic.add.flt.1d", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2206, intr_name="llvm.amdgcn.image.atomic.add.flt.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2207, intr_name="llvm.amdgcn.image.atomic.add.flt.2d", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2208, intr_name="llvm.amdgcn.image.atomic.add.flt.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2209, intr_name="llvm.amdgcn.image.atomic.add.flt.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2210, intr_name="llvm.amdgcn.image.atomic.add.flt.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2211, intr_name="llvm.amdgcn.image.atomic.add.flt.3d", is_overloaded=True, return_type=return_type)


def image_atomic_add_flt_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2212, intr_name="llvm.amdgcn.image.atomic.add.flt.cube", is_overloaded=True, return_type=return_type)


def image_atomic_and_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2213, intr_name="llvm.amdgcn.image.atomic.and.1d", is_overloaded=True, return_type=return_type)


def image_atomic_and_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2214, intr_name="llvm.amdgcn.image.atomic.and.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_and_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2215, intr_name="llvm.amdgcn.image.atomic.and.2d", is_overloaded=True, return_type=return_type)


def image_atomic_and_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2216, intr_name="llvm.amdgcn.image.atomic.and.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_and_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2217, intr_name="llvm.amdgcn.image.atomic.and.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_and_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2218, intr_name="llvm.amdgcn.image.atomic.and.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_and_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2219, intr_name="llvm.amdgcn.image.atomic.and.3d", is_overloaded=True, return_type=return_type)


def image_atomic_and_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2220, intr_name="llvm.amdgcn.image.atomic.and.cube", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_1d(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2221, intr_name="llvm.amdgcn.image.atomic.cmpswap.1d", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_1darray(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2222, intr_name="llvm.amdgcn.image.atomic.cmpswap.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_2d(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2223, intr_name="llvm.amdgcn.image.atomic.cmpswap.2d", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_2darray(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2224, intr_name="llvm.amdgcn.image.atomic.cmpswap.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_2darraymsaa(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2225, intr_name="llvm.amdgcn.image.atomic.cmpswap.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_2dmsaa(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2226, intr_name="llvm.amdgcn.image.atomic.cmpswap.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_3d(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2227, intr_name="llvm.amdgcn.image.atomic.cmpswap.3d", is_overloaded=True, return_type=return_type)


def image_atomic_cmpswap_cube(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2228, intr_name="llvm.amdgcn.image.atomic.cmpswap.cube", is_overloaded=True, return_type=return_type)


def image_atomic_dec_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2229, intr_name="llvm.amdgcn.image.atomic.dec.1d", is_overloaded=True, return_type=return_type)


def image_atomic_dec_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2230, intr_name="llvm.amdgcn.image.atomic.dec.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_dec_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2231, intr_name="llvm.amdgcn.image.atomic.dec.2d", is_overloaded=True, return_type=return_type)


def image_atomic_dec_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2232, intr_name="llvm.amdgcn.image.atomic.dec.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_dec_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2233, intr_name="llvm.amdgcn.image.atomic.dec.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_dec_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2234, intr_name="llvm.amdgcn.image.atomic.dec.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_dec_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2235, intr_name="llvm.amdgcn.image.atomic.dec.3d", is_overloaded=True, return_type=return_type)


def image_atomic_dec_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2236, intr_name="llvm.amdgcn.image.atomic.dec.cube", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2237, intr_name="llvm.amdgcn.image.atomic.fmax.1d", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2238, intr_name="llvm.amdgcn.image.atomic.fmax.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2239, intr_name="llvm.amdgcn.image.atomic.fmax.2d", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2240, intr_name="llvm.amdgcn.image.atomic.fmax.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2241, intr_name="llvm.amdgcn.image.atomic.fmax.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2242, intr_name="llvm.amdgcn.image.atomic.fmax.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2243, intr_name="llvm.amdgcn.image.atomic.fmax.3d", is_overloaded=True, return_type=return_type)


def image_atomic_fmax_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2244, intr_name="llvm.amdgcn.image.atomic.fmax.cube", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2245, intr_name="llvm.amdgcn.image.atomic.fmin.1d", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2246, intr_name="llvm.amdgcn.image.atomic.fmin.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2247, intr_name="llvm.amdgcn.image.atomic.fmin.2d", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2248, intr_name="llvm.amdgcn.image.atomic.fmin.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2249, intr_name="llvm.amdgcn.image.atomic.fmin.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2250, intr_name="llvm.amdgcn.image.atomic.fmin.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2251, intr_name="llvm.amdgcn.image.atomic.fmin.3d", is_overloaded=True, return_type=return_type)


def image_atomic_fmin_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2252, intr_name="llvm.amdgcn.image.atomic.fmin.cube", is_overloaded=True, return_type=return_type)


def image_atomic_inc_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2253, intr_name="llvm.amdgcn.image.atomic.inc.1d", is_overloaded=True, return_type=return_type)


def image_atomic_inc_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2254, intr_name="llvm.amdgcn.image.atomic.inc.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_inc_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2255, intr_name="llvm.amdgcn.image.atomic.inc.2d", is_overloaded=True, return_type=return_type)


def image_atomic_inc_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2256, intr_name="llvm.amdgcn.image.atomic.inc.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_inc_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2257, intr_name="llvm.amdgcn.image.atomic.inc.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_inc_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2258, intr_name="llvm.amdgcn.image.atomic.inc.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_inc_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2259, intr_name="llvm.amdgcn.image.atomic.inc.3d", is_overloaded=True, return_type=return_type)


def image_atomic_inc_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2260, intr_name="llvm.amdgcn.image.atomic.inc.cube", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2261, intr_name="llvm.amdgcn.image.atomic.max.flt.1d", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2262, intr_name="llvm.amdgcn.image.atomic.max.flt.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2263, intr_name="llvm.amdgcn.image.atomic.max.flt.2d", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2264, intr_name="llvm.amdgcn.image.atomic.max.flt.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2265, intr_name="llvm.amdgcn.image.atomic.max.flt.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2266, intr_name="llvm.amdgcn.image.atomic.max.flt.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2267, intr_name="llvm.amdgcn.image.atomic.max.flt.3d", is_overloaded=True, return_type=return_type)


def image_atomic_max_flt_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2268, intr_name="llvm.amdgcn.image.atomic.max.flt.cube", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2269, intr_name="llvm.amdgcn.image.atomic.min.flt.1d", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2270, intr_name="llvm.amdgcn.image.atomic.min.flt.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2271, intr_name="llvm.amdgcn.image.atomic.min.flt.2d", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2272, intr_name="llvm.amdgcn.image.atomic.min.flt.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2273, intr_name="llvm.amdgcn.image.atomic.min.flt.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2274, intr_name="llvm.amdgcn.image.atomic.min.flt.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2275, intr_name="llvm.amdgcn.image.atomic.min.flt.3d", is_overloaded=True, return_type=return_type)


def image_atomic_min_flt_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2276, intr_name="llvm.amdgcn.image.atomic.min.flt.cube", is_overloaded=True, return_type=return_type)


def image_atomic_or_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2277, intr_name="llvm.amdgcn.image.atomic.or.1d", is_overloaded=True, return_type=return_type)


def image_atomic_or_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2278, intr_name="llvm.amdgcn.image.atomic.or.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_or_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2279, intr_name="llvm.amdgcn.image.atomic.or.2d", is_overloaded=True, return_type=return_type)


def image_atomic_or_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2280, intr_name="llvm.amdgcn.image.atomic.or.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_or_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2281, intr_name="llvm.amdgcn.image.atomic.or.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_or_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2282, intr_name="llvm.amdgcn.image.atomic.or.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_or_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2283, intr_name="llvm.amdgcn.image.atomic.or.3d", is_overloaded=True, return_type=return_type)


def image_atomic_or_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2284, intr_name="llvm.amdgcn.image.atomic.or.cube", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2285, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.1d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2286, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2287, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.2d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2288, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2289, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2290, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2291, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.3d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_bf16_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2292, intr_name="llvm.amdgcn.image.atomic.pk.add.bf16.cube", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2293, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.1d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2294, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2295, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.2d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2296, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2297, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2298, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2299, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.3d", is_overloaded=True, return_type=return_type)


def image_atomic_pk_add_f16_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2300, intr_name="llvm.amdgcn.image.atomic.pk.add.f16.cube", is_overloaded=True, return_type=return_type)


def image_atomic_smax_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2301, intr_name="llvm.amdgcn.image.atomic.smax.1d", is_overloaded=True, return_type=return_type)


def image_atomic_smax_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2302, intr_name="llvm.amdgcn.image.atomic.smax.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_smax_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2303, intr_name="llvm.amdgcn.image.atomic.smax.2d", is_overloaded=True, return_type=return_type)


def image_atomic_smax_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2304, intr_name="llvm.amdgcn.image.atomic.smax.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_smax_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2305, intr_name="llvm.amdgcn.image.atomic.smax.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_smax_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2306, intr_name="llvm.amdgcn.image.atomic.smax.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_smax_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2307, intr_name="llvm.amdgcn.image.atomic.smax.3d", is_overloaded=True, return_type=return_type)


def image_atomic_smax_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2308, intr_name="llvm.amdgcn.image.atomic.smax.cube", is_overloaded=True, return_type=return_type)


def image_atomic_smin_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2309, intr_name="llvm.amdgcn.image.atomic.smin.1d", is_overloaded=True, return_type=return_type)


def image_atomic_smin_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2310, intr_name="llvm.amdgcn.image.atomic.smin.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_smin_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2311, intr_name="llvm.amdgcn.image.atomic.smin.2d", is_overloaded=True, return_type=return_type)


def image_atomic_smin_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2312, intr_name="llvm.amdgcn.image.atomic.smin.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_smin_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2313, intr_name="llvm.amdgcn.image.atomic.smin.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_smin_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2314, intr_name="llvm.amdgcn.image.atomic.smin.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_smin_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2315, intr_name="llvm.amdgcn.image.atomic.smin.3d", is_overloaded=True, return_type=return_type)


def image_atomic_smin_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2316, intr_name="llvm.amdgcn.image.atomic.smin.cube", is_overloaded=True, return_type=return_type)


def image_atomic_sub_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2317, intr_name="llvm.amdgcn.image.atomic.sub.1d", is_overloaded=True, return_type=return_type)


def image_atomic_sub_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2318, intr_name="llvm.amdgcn.image.atomic.sub.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_sub_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2319, intr_name="llvm.amdgcn.image.atomic.sub.2d", is_overloaded=True, return_type=return_type)


def image_atomic_sub_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2320, intr_name="llvm.amdgcn.image.atomic.sub.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_sub_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2321, intr_name="llvm.amdgcn.image.atomic.sub.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_sub_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2322, intr_name="llvm.amdgcn.image.atomic.sub.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_sub_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2323, intr_name="llvm.amdgcn.image.atomic.sub.3d", is_overloaded=True, return_type=return_type)


def image_atomic_sub_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2324, intr_name="llvm.amdgcn.image.atomic.sub.cube", is_overloaded=True, return_type=return_type)


def image_atomic_swap_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2325, intr_name="llvm.amdgcn.image.atomic.swap.1d", is_overloaded=True, return_type=return_type)


def image_atomic_swap_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2326, intr_name="llvm.amdgcn.image.atomic.swap.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_swap_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2327, intr_name="llvm.amdgcn.image.atomic.swap.2d", is_overloaded=True, return_type=return_type)


def image_atomic_swap_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2328, intr_name="llvm.amdgcn.image.atomic.swap.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_swap_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2329, intr_name="llvm.amdgcn.image.atomic.swap.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_swap_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2330, intr_name="llvm.amdgcn.image.atomic.swap.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_swap_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2331, intr_name="llvm.amdgcn.image.atomic.swap.3d", is_overloaded=True, return_type=return_type)


def image_atomic_swap_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2332, intr_name="llvm.amdgcn.image.atomic.swap.cube", is_overloaded=True, return_type=return_type)


def image_atomic_umax_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2333, intr_name="llvm.amdgcn.image.atomic.umax.1d", is_overloaded=True, return_type=return_type)


def image_atomic_umax_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2334, intr_name="llvm.amdgcn.image.atomic.umax.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_umax_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2335, intr_name="llvm.amdgcn.image.atomic.umax.2d", is_overloaded=True, return_type=return_type)


def image_atomic_umax_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2336, intr_name="llvm.amdgcn.image.atomic.umax.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_umax_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2337, intr_name="llvm.amdgcn.image.atomic.umax.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_umax_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2338, intr_name="llvm.amdgcn.image.atomic.umax.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_umax_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2339, intr_name="llvm.amdgcn.image.atomic.umax.3d", is_overloaded=True, return_type=return_type)


def image_atomic_umax_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2340, intr_name="llvm.amdgcn.image.atomic.umax.cube", is_overloaded=True, return_type=return_type)


def image_atomic_umin_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2341, intr_name="llvm.amdgcn.image.atomic.umin.1d", is_overloaded=True, return_type=return_type)


def image_atomic_umin_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2342, intr_name="llvm.amdgcn.image.atomic.umin.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_umin_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2343, intr_name="llvm.amdgcn.image.atomic.umin.2d", is_overloaded=True, return_type=return_type)


def image_atomic_umin_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2344, intr_name="llvm.amdgcn.image.atomic.umin.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_umin_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2345, intr_name="llvm.amdgcn.image.atomic.umin.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_umin_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2346, intr_name="llvm.amdgcn.image.atomic.umin.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_umin_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2347, intr_name="llvm.amdgcn.image.atomic.umin.3d", is_overloaded=True, return_type=return_type)


def image_atomic_umin_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2348, intr_name="llvm.amdgcn.image.atomic.umin.cube", is_overloaded=True, return_type=return_type)


def image_atomic_xor_1d(a: LLVMMatchType[Literal[0]], b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2349, intr_name="llvm.amdgcn.image.atomic.xor.1d", is_overloaded=True, return_type=return_type)


def image_atomic_xor_1darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2350, intr_name="llvm.amdgcn.image.atomic.xor.1darray", is_overloaded=True, return_type=return_type)


def image_atomic_xor_2d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2351, intr_name="llvm.amdgcn.image.atomic.xor.2d", is_overloaded=True, return_type=return_type)


def image_atomic_xor_2darray(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2352, intr_name="llvm.amdgcn.image.atomic.xor.2darray", is_overloaded=True, return_type=return_type)


def image_atomic_xor_2darraymsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2353, intr_name="llvm.amdgcn.image.atomic.xor.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_atomic_xor_2dmsaa(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2354, intr_name="llvm.amdgcn.image.atomic.xor.2dmsaa", is_overloaded=True, return_type=return_type)


def image_atomic_xor_3d(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2355, intr_name="llvm.amdgcn.image.atomic.xor.3d", is_overloaded=True, return_type=return_type)


def image_atomic_xor_cube(a: LLVMMatchType[Literal[0]], b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2356, intr_name="llvm.amdgcn.image.atomic.xor.cube", is_overloaded=True, return_type=return_type)


def image_bvh_intersect_ray(a: anyint, b: float, c: v3f32, d: anyvector, e: LLVMMatchType[Literal[1]], f: v4i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2357, intr_name="llvm.amdgcn.image.bvh.intersect.ray", is_overloaded=True, return_type=return_type)


def image_gather4_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2358, intr_name="llvm.amdgcn.image.gather4.2d", is_overloaded=True, return_type=return_type)


def image_gather4_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2359, intr_name="llvm.amdgcn.image.gather4.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_b_2d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2360, intr_name="llvm.amdgcn.image.gather4.b.2d", is_overloaded=True, return_type=return_type)


def image_gather4_b_2darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2361, intr_name="llvm.amdgcn.image.gather4.b.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_2d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2362, intr_name="llvm.amdgcn.image.gather4.b.cl.2d", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_2darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2363, intr_name="llvm.amdgcn.image.gather4.b.cl.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_cube(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2364, intr_name="llvm.amdgcn.image.gather4.b.cl.cube", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_o_2d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2365, intr_name="llvm.amdgcn.image.gather4.b.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2366, intr_name="llvm.amdgcn.image.gather4.b.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_b_cl_o_cube(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2367, intr_name="llvm.amdgcn.image.gather4.b.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_b_cube(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2368, intr_name="llvm.amdgcn.image.gather4.b.cube", is_overloaded=True, return_type=return_type)


def image_gather4_b_o_2d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2369, intr_name="llvm.amdgcn.image.gather4.b.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_b_o_2darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2370, intr_name="llvm.amdgcn.image.gather4.b.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_b_o_cube(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2371, intr_name="llvm.amdgcn.image.gather4.b.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2372, intr_name="llvm.amdgcn.image.gather4.c.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2373, intr_name="llvm.amdgcn.image.gather4.c.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_2d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2374, intr_name="llvm.amdgcn.image.gather4.c.b.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_2darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2375, intr_name="llvm.amdgcn.image.gather4.c.b.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_2d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2376, intr_name="llvm.amdgcn.image.gather4.c.b.cl.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_2darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2377, intr_name="llvm.amdgcn.image.gather4.c.b.cl.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_cube(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2378, intr_name="llvm.amdgcn.image.gather4.c.b.cl.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_o_2d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2379, intr_name="llvm.amdgcn.image.gather4.c.b.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2380, intr_name="llvm.amdgcn.image.gather4.c.b.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cl_o_cube(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2381, intr_name="llvm.amdgcn.image.gather4.c.b.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_cube(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2382, intr_name="llvm.amdgcn.image.gather4.c.b.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_o_2d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2383, intr_name="llvm.amdgcn.image.gather4.c.b.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_o_2darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2384, intr_name="llvm.amdgcn.image.gather4.c.b.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_b_o_cube(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2385, intr_name="llvm.amdgcn.image.gather4.c.b.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2386, intr_name="llvm.amdgcn.image.gather4.c.cl.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2387, intr_name="llvm.amdgcn.image.gather4.c.cl.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2388, intr_name="llvm.amdgcn.image.gather4.c.cl.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2389, intr_name="llvm.amdgcn.image.gather4.c.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2390, intr_name="llvm.amdgcn.image.gather4.c.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_cl_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2391, intr_name="llvm.amdgcn.image.gather4.c.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2392, intr_name="llvm.amdgcn.image.gather4.c.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2393, intr_name="llvm.amdgcn.image.gather4.c.l.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2394, intr_name="llvm.amdgcn.image.gather4.c.l.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2395, intr_name="llvm.amdgcn.image.gather4.c.l.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2396, intr_name="llvm.amdgcn.image.gather4.c.l.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2397, intr_name="llvm.amdgcn.image.gather4.c.l.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_l_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2398, intr_name="llvm.amdgcn.image.gather4.c.l.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2399, intr_name="llvm.amdgcn.image.gather4.c.lz.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2400, intr_name="llvm.amdgcn.image.gather4.c.lz.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2401, intr_name="llvm.amdgcn.image.gather4.c.lz.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2402, intr_name="llvm.amdgcn.image.gather4.c.lz.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2403, intr_name="llvm.amdgcn.image.gather4.c.lz.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_lz_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2404, intr_name="llvm.amdgcn.image.gather4.c.lz.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_c_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2405, intr_name="llvm.amdgcn.image.gather4.c.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_c_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2406, intr_name="llvm.amdgcn.image.gather4.c.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_c_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2407, intr_name="llvm.amdgcn.image.gather4.c.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_cl_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2408, intr_name="llvm.amdgcn.image.gather4.cl.2d", is_overloaded=True, return_type=return_type)


def image_gather4_cl_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2409, intr_name="llvm.amdgcn.image.gather4.cl.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_cl_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2410, intr_name="llvm.amdgcn.image.gather4.cl.cube", is_overloaded=True, return_type=return_type)


def image_gather4_cl_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2411, intr_name="llvm.amdgcn.image.gather4.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2412, intr_name="llvm.amdgcn.image.gather4.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_cl_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2413, intr_name="llvm.amdgcn.image.gather4.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2414, intr_name="llvm.amdgcn.image.gather4.cube", is_overloaded=True, return_type=return_type)


def image_gather4_l_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2415, intr_name="llvm.amdgcn.image.gather4.l.2d", is_overloaded=True, return_type=return_type)


def image_gather4_l_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2416, intr_name="llvm.amdgcn.image.gather4.l.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_l_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2417, intr_name="llvm.amdgcn.image.gather4.l.cube", is_overloaded=True, return_type=return_type)


def image_gather4_l_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2418, intr_name="llvm.amdgcn.image.gather4.l.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_l_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2419, intr_name="llvm.amdgcn.image.gather4.l.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_l_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2420, intr_name="llvm.amdgcn.image.gather4.l.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_lz_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2421, intr_name="llvm.amdgcn.image.gather4.lz.2d", is_overloaded=True, return_type=return_type)


def image_gather4_lz_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2422, intr_name="llvm.amdgcn.image.gather4.lz.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_lz_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2423, intr_name="llvm.amdgcn.image.gather4.lz.cube", is_overloaded=True, return_type=return_type)


def image_gather4_lz_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2424, intr_name="llvm.amdgcn.image.gather4.lz.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_lz_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2425, intr_name="llvm.amdgcn.image.gather4.lz.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_lz_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2426, intr_name="llvm.amdgcn.image.gather4.lz.o.cube", is_overloaded=True, return_type=return_type)


def image_gather4_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2427, intr_name="llvm.amdgcn.image.gather4.o.2d", is_overloaded=True, return_type=return_type)


def image_gather4_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2428, intr_name="llvm.amdgcn.image.gather4.o.2darray", is_overloaded=True, return_type=return_type)


def image_gather4_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2429, intr_name="llvm.amdgcn.image.gather4.o.cube", is_overloaded=True, return_type=return_type)


def image_getlod_1d(a: i32, b: anyfloat, c: any, d: any, e: i1, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2430, intr_name="llvm.amdgcn.image.getlod.1d", is_overloaded=True, return_type=return_type)


def image_getlod_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2431, intr_name="llvm.amdgcn.image.getlod.1darray", is_overloaded=True, return_type=return_type)


def image_getlod_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2432, intr_name="llvm.amdgcn.image.getlod.2d", is_overloaded=True, return_type=return_type)


def image_getlod_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2433, intr_name="llvm.amdgcn.image.getlod.2darray", is_overloaded=True, return_type=return_type)


def image_getlod_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2434, intr_name="llvm.amdgcn.image.getlod.3d", is_overloaded=True, return_type=return_type)


def image_getlod_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2435, intr_name="llvm.amdgcn.image.getlod.cube", is_overloaded=True, return_type=return_type)


def image_getresinfo_1d(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2436, intr_name="llvm.amdgcn.image.getresinfo.1d", is_overloaded=True, return_type=return_type)


def image_getresinfo_1darray(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2437, intr_name="llvm.amdgcn.image.getresinfo.1darray", is_overloaded=True, return_type=return_type)


def image_getresinfo_2d(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2438, intr_name="llvm.amdgcn.image.getresinfo.2d", is_overloaded=True, return_type=return_type)


def image_getresinfo_2darray(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2439, intr_name="llvm.amdgcn.image.getresinfo.2darray", is_overloaded=True, return_type=return_type)


def image_getresinfo_2darraymsaa(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2440, intr_name="llvm.amdgcn.image.getresinfo.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_getresinfo_2dmsaa(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2441, intr_name="llvm.amdgcn.image.getresinfo.2dmsaa", is_overloaded=True, return_type=return_type)


def image_getresinfo_3d(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2442, intr_name="llvm.amdgcn.image.getresinfo.3d", is_overloaded=True, return_type=return_type)


def image_getresinfo_cube(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2443, intr_name="llvm.amdgcn.image.getresinfo.cube", is_overloaded=True, return_type=return_type)


def image_load_1d(a: i32, b: anyint, c: any, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2444, intr_name="llvm.amdgcn.image.load.1d", is_overloaded=True, return_type=return_type)


def image_load_1darray(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2445, intr_name="llvm.amdgcn.image.load.1darray", is_overloaded=True, return_type=return_type)


def image_load_2d(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2446, intr_name="llvm.amdgcn.image.load.2d", is_overloaded=True, return_type=return_type)


def image_load_2darray(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2447, intr_name="llvm.amdgcn.image.load.2darray", is_overloaded=True, return_type=return_type)


def image_load_2darraymsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2448, intr_name="llvm.amdgcn.image.load.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_load_2dmsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2449, intr_name="llvm.amdgcn.image.load.2dmsaa", is_overloaded=True, return_type=return_type)


def image_load_3d(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2450, intr_name="llvm.amdgcn.image.load.3d", is_overloaded=True, return_type=return_type)


def image_load_cube(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2451, intr_name="llvm.amdgcn.image.load.cube", is_overloaded=True, return_type=return_type)


def image_load_mip_1d(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: any, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2452, intr_name="llvm.amdgcn.image.load.mip.1d", is_overloaded=True, return_type=return_type)


def image_load_mip_1darray(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2453, intr_name="llvm.amdgcn.image.load.mip.1darray", is_overloaded=True, return_type=return_type)


def image_load_mip_2d(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2454, intr_name="llvm.amdgcn.image.load.mip.2d", is_overloaded=True, return_type=return_type)


def image_load_mip_2darray(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2455, intr_name="llvm.amdgcn.image.load.mip.2darray", is_overloaded=True, return_type=return_type)


def image_load_mip_3d(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2456, intr_name="llvm.amdgcn.image.load.mip.3d", is_overloaded=True, return_type=return_type)


def image_load_mip_cube(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2457, intr_name="llvm.amdgcn.image.load.mip.cube", is_overloaded=True, return_type=return_type)


def image_msaa_load_2darraymsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2458, intr_name="llvm.amdgcn.image.msaa.load.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_msaa_load_2dmsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2459, intr_name="llvm.amdgcn.image.msaa.load.2dmsaa", is_overloaded=True, return_type=return_type)


def image_msaa_load_x_2darraymsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2460, intr_name="llvm.amdgcn.image.msaa.load.x.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_msaa_load_x_2dmsaa(a: i32, b: anyint, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2461, intr_name="llvm.amdgcn.image.msaa.load.x.2dmsaa", is_overloaded=True, return_type=return_type)


def image_sample_1d(a: i32, b: anyfloat, c: any, d: any, e: i1, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2462, intr_name="llvm.amdgcn.image.sample.1d", is_overloaded=True, return_type=return_type)


def image_sample_1d_nortn(a: i32, b: anyfloat, c: any, d: any, e: i1, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=2463, intr_name="llvm.amdgcn.image.sample.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2464, intr_name="llvm.amdgcn.image.sample.1darray", is_overloaded=True, return_type=return_type)


def image_sample_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2465, intr_name="llvm.amdgcn.image.sample.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2466, intr_name="llvm.amdgcn.image.sample.2d", is_overloaded=True, return_type=return_type)


def image_sample_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2467, intr_name="llvm.amdgcn.image.sample.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2468, intr_name="llvm.amdgcn.image.sample.2darray", is_overloaded=True, return_type=return_type)


def image_sample_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2469, intr_name="llvm.amdgcn.image.sample.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2470, intr_name="llvm.amdgcn.image.sample.3d", is_overloaded=True, return_type=return_type)


def image_sample_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2471, intr_name="llvm.amdgcn.image.sample.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_1d(a: i32, b: anyfloat, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2472, intr_name="llvm.amdgcn.image.sample.b.1d", is_overloaded=True, return_type=return_type)


def image_sample_b_1d_nortn(a: i32, b: anyfloat, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2473, intr_name="llvm.amdgcn.image.sample.b.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_1darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2474, intr_name="llvm.amdgcn.image.sample.b.1darray", is_overloaded=True, return_type=return_type)


def image_sample_b_1darray_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2475, intr_name="llvm.amdgcn.image.sample.b.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_2d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2476, intr_name="llvm.amdgcn.image.sample.b.2d", is_overloaded=True, return_type=return_type)


def image_sample_b_2d_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2477, intr_name="llvm.amdgcn.image.sample.b.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_2darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2478, intr_name="llvm.amdgcn.image.sample.b.2darray", is_overloaded=True, return_type=return_type)


def image_sample_b_2darray_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2479, intr_name="llvm.amdgcn.image.sample.b.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_3d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2480, intr_name="llvm.amdgcn.image.sample.b.3d", is_overloaded=True, return_type=return_type)


def image_sample_b_3d_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2481, intr_name="llvm.amdgcn.image.sample.b.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_1d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2482, intr_name="llvm.amdgcn.image.sample.b.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_1d_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2483, intr_name="llvm.amdgcn.image.sample.b.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_1darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2484, intr_name="llvm.amdgcn.image.sample.b.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_1darray_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2485, intr_name="llvm.amdgcn.image.sample.b.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_2d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2486, intr_name="llvm.amdgcn.image.sample.b.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_2d_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2487, intr_name="llvm.amdgcn.image.sample.b.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_2darray(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2488, intr_name="llvm.amdgcn.image.sample.b.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_2darray_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2489, intr_name="llvm.amdgcn.image.sample.b.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_3d(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2490, intr_name="llvm.amdgcn.image.sample.b.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_3d_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2491, intr_name="llvm.amdgcn.image.sample.b.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_cube(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2492, intr_name="llvm.amdgcn.image.sample.b.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_cube_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2493, intr_name="llvm.amdgcn.image.sample.b.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_1d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2494, intr_name="llvm.amdgcn.image.sample.b.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2495, intr_name="llvm.amdgcn.image.sample.b.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_1darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2496, intr_name="llvm.amdgcn.image.sample.b.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2497, intr_name="llvm.amdgcn.image.sample.b.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_2d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2498, intr_name="llvm.amdgcn.image.sample.b.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2499, intr_name="llvm.amdgcn.image.sample.b.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2500, intr_name="llvm.amdgcn.image.sample.b.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2501, intr_name="llvm.amdgcn.image.sample.b.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_3d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2502, intr_name="llvm.amdgcn.image.sample.b.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2503, intr_name="llvm.amdgcn.image.sample.b.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_cube(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2504, intr_name="llvm.amdgcn.image.sample.b.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_b_cl_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2505, intr_name="llvm.amdgcn.image.sample.b.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_cube(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[2]], e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2506, intr_name="llvm.amdgcn.image.sample.b.cube", is_overloaded=True, return_type=return_type)


def image_sample_b_cube_nortn(a: i32, b: anyfloat, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2507, intr_name="llvm.amdgcn.image.sample.b.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_1d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2508, intr_name="llvm.amdgcn.image.sample.b.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_b_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2509, intr_name="llvm.amdgcn.image.sample.b.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_1darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2510, intr_name="llvm.amdgcn.image.sample.b.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_b_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2511, intr_name="llvm.amdgcn.image.sample.b.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_2d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2512, intr_name="llvm.amdgcn.image.sample.b.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_b_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2513, intr_name="llvm.amdgcn.image.sample.b.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_2darray(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2514, intr_name="llvm.amdgcn.image.sample.b.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_b_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2515, intr_name="llvm.amdgcn.image.sample.b.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_3d(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2516, intr_name="llvm.amdgcn.image.sample.b.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_b_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2517, intr_name="llvm.amdgcn.image.sample.b.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_b_o_cube(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2518, intr_name="llvm.amdgcn.image.sample.b.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_b_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2519, intr_name="llvm.amdgcn.image.sample.b.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_1d(a: i32, b: float, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2520, intr_name="llvm.amdgcn.image.sample.c.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_1d_nortn(a: i32, b: float, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2521, intr_name="llvm.amdgcn.image.sample.c.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2522, intr_name="llvm.amdgcn.image.sample.c.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2523, intr_name="llvm.amdgcn.image.sample.c.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2524, intr_name="llvm.amdgcn.image.sample.c.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2525, intr_name="llvm.amdgcn.image.sample.c.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2526, intr_name="llvm.amdgcn.image.sample.c.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2527, intr_name="llvm.amdgcn.image.sample.c.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2528, intr_name="llvm.amdgcn.image.sample.c.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2529, intr_name="llvm.amdgcn.image.sample.c.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_1d(a: i32, b: anyfloat, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2530, intr_name="llvm.amdgcn.image.sample.c.b.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_1d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2531, intr_name="llvm.amdgcn.image.sample.c.b.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_1darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2532, intr_name="llvm.amdgcn.image.sample.c.b.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_1darray_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2533, intr_name="llvm.amdgcn.image.sample.c.b.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_2d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2534, intr_name="llvm.amdgcn.image.sample.c.b.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_2d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2535, intr_name="llvm.amdgcn.image.sample.c.b.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_2darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2536, intr_name="llvm.amdgcn.image.sample.c.b.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_2darray_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2537, intr_name="llvm.amdgcn.image.sample.c.b.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_3d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2538, intr_name="llvm.amdgcn.image.sample.c.b.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_3d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2539, intr_name="llvm.amdgcn.image.sample.c.b.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_1d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2540, intr_name="llvm.amdgcn.image.sample.c.b.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_1d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2541, intr_name="llvm.amdgcn.image.sample.c.b.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_1darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2542, intr_name="llvm.amdgcn.image.sample.c.b.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_1darray_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2543, intr_name="llvm.amdgcn.image.sample.c.b.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_2d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2544, intr_name="llvm.amdgcn.image.sample.c.b.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_2d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2545, intr_name="llvm.amdgcn.image.sample.c.b.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_2darray(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2546, intr_name="llvm.amdgcn.image.sample.c.b.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_2darray_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2547, intr_name="llvm.amdgcn.image.sample.c.b.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_3d(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2548, intr_name="llvm.amdgcn.image.sample.c.b.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_3d_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2549, intr_name="llvm.amdgcn.image.sample.c.b.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_cube(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2550, intr_name="llvm.amdgcn.image.sample.c.b.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_cube_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2551, intr_name="llvm.amdgcn.image.sample.c.b.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_1d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2552, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2553, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_1darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2554, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2555, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_2d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2556, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2557, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2558, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2559, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_3d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2560, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2561, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_cube(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2562, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cl_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2563, intr_name="llvm.amdgcn.image.sample.c.b.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cube(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2564, intr_name="llvm.amdgcn.image.sample.c.b.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_b_cube_nortn(a: i32, b: anyfloat, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2565, intr_name="llvm.amdgcn.image.sample.c.b.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_1d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2566, intr_name="llvm.amdgcn.image.sample.c.b.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2567, intr_name="llvm.amdgcn.image.sample.c.b.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_1darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2568, intr_name="llvm.amdgcn.image.sample.c.b.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2569, intr_name="llvm.amdgcn.image.sample.c.b.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_2d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2570, intr_name="llvm.amdgcn.image.sample.c.b.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2571, intr_name="llvm.amdgcn.image.sample.c.b.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_2darray(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2572, intr_name="llvm.amdgcn.image.sample.c.b.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2573, intr_name="llvm.amdgcn.image.sample.c.b.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_3d(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2574, intr_name="llvm.amdgcn.image.sample.c.b.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2575, intr_name="llvm.amdgcn.image.sample.c.b.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_cube(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2576, intr_name="llvm.amdgcn.image.sample.c.b.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_b_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: float, e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2577, intr_name="llvm.amdgcn.image.sample.c.b.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2578, intr_name="llvm.amdgcn.image.sample.c.cd.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2579, intr_name="llvm.amdgcn.image.sample.c.cd.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2580, intr_name="llvm.amdgcn.image.sample.c.cd.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2581, intr_name="llvm.amdgcn.image.sample.c.cd.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2582, intr_name="llvm.amdgcn.image.sample.c.cd.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2583, intr_name="llvm.amdgcn.image.sample.c.cd.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2584, intr_name="llvm.amdgcn.image.sample.c.cd.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2585, intr_name="llvm.amdgcn.image.sample.c.cd.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2586, intr_name="llvm.amdgcn.image.sample.c.cd.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2587, intr_name="llvm.amdgcn.image.sample.c.cd.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2588, intr_name="llvm.amdgcn.image.sample.c.cd.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2589, intr_name="llvm.amdgcn.image.sample.c.cd.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2590, intr_name="llvm.amdgcn.image.sample.c.cd.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2591, intr_name="llvm.amdgcn.image.sample.c.cd.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2592, intr_name="llvm.amdgcn.image.sample.c.cd.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2593, intr_name="llvm.amdgcn.image.sample.c.cd.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2594, intr_name="llvm.amdgcn.image.sample.c.cd.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2595, intr_name="llvm.amdgcn.image.sample.c.cd.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2596, intr_name="llvm.amdgcn.image.sample.c.cd.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2597, intr_name="llvm.amdgcn.image.sample.c.cd.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2598, intr_name="llvm.amdgcn.image.sample.c.cd.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2599, intr_name="llvm.amdgcn.image.sample.c.cd.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2600, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2601, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2602, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2603, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2604, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2605, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2606, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2607, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: anyfloat, k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: LLVMMatchType[Literal[2]], n: any, o: any, p: i1, q: i32, r: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, intr_id=2608, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: LLVMMatchType[Literal[0]], j: anyfloat, k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: LLVMMatchType[Literal[1]], n: any, o: any, p: i1, q: i32, r: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, intr_id=2609, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2610, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cl_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2611, intr_name="llvm.amdgcn.image.sample.c.cd.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2612, intr_name="llvm.amdgcn.image.sample.c.cd.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2613, intr_name="llvm.amdgcn.image.sample.c.cd.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2614, intr_name="llvm.amdgcn.image.sample.c.cd.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2615, intr_name="llvm.amdgcn.image.sample.c.cd.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2616, intr_name="llvm.amdgcn.image.sample.c.cd.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2617, intr_name="llvm.amdgcn.image.sample.c.cd.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2618, intr_name="llvm.amdgcn.image.sample.c.cd.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2619, intr_name="llvm.amdgcn.image.sample.c.cd.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2620, intr_name="llvm.amdgcn.image.sample.c.cd.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2621, intr_name="llvm.amdgcn.image.sample.c.cd.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: anyfloat, k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2622, intr_name="llvm.amdgcn.image.sample.c.cd.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: LLVMMatchType[Literal[0]], j: anyfloat, k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2623, intr_name="llvm.amdgcn.image.sample.c.cd.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2624, intr_name="llvm.amdgcn.image.sample.c.cd.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cd_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2625, intr_name="llvm.amdgcn.image.sample.c.cd.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2626, intr_name="llvm.amdgcn.image.sample.c.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2627, intr_name="llvm.amdgcn.image.sample.c.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2628, intr_name="llvm.amdgcn.image.sample.c.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2629, intr_name="llvm.amdgcn.image.sample.c.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2630, intr_name="llvm.amdgcn.image.sample.c.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2631, intr_name="llvm.amdgcn.image.sample.c.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2632, intr_name="llvm.amdgcn.image.sample.c.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2633, intr_name="llvm.amdgcn.image.sample.c.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2634, intr_name="llvm.amdgcn.image.sample.c.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2635, intr_name="llvm.amdgcn.image.sample.c.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2636, intr_name="llvm.amdgcn.image.sample.c.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2637, intr_name="llvm.amdgcn.image.sample.c.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2638, intr_name="llvm.amdgcn.image.sample.c.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2639, intr_name="llvm.amdgcn.image.sample.c.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2640, intr_name="llvm.amdgcn.image.sample.c.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2641, intr_name="llvm.amdgcn.image.sample.c.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2642, intr_name="llvm.amdgcn.image.sample.c.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2643, intr_name="llvm.amdgcn.image.sample.c.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2644, intr_name="llvm.amdgcn.image.sample.c.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2645, intr_name="llvm.amdgcn.image.sample.c.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2646, intr_name="llvm.amdgcn.image.sample.c.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2647, intr_name="llvm.amdgcn.image.sample.c.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2648, intr_name="llvm.amdgcn.image.sample.c.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cl_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2649, intr_name="llvm.amdgcn.image.sample.c.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2650, intr_name="llvm.amdgcn.image.sample.c.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2651, intr_name="llvm.amdgcn.image.sample.c.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2652, intr_name="llvm.amdgcn.image.sample.c.d.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2653, intr_name="llvm.amdgcn.image.sample.c.d.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2654, intr_name="llvm.amdgcn.image.sample.c.d.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2655, intr_name="llvm.amdgcn.image.sample.c.d.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2656, intr_name="llvm.amdgcn.image.sample.c.d.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2657, intr_name="llvm.amdgcn.image.sample.c.d.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2658, intr_name="llvm.amdgcn.image.sample.c.d.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2659, intr_name="llvm.amdgcn.image.sample.c.d.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2660, intr_name="llvm.amdgcn.image.sample.c.d.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2661, intr_name="llvm.amdgcn.image.sample.c.d.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2662, intr_name="llvm.amdgcn.image.sample.c.d.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2663, intr_name="llvm.amdgcn.image.sample.c.d.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2664, intr_name="llvm.amdgcn.image.sample.c.d.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2665, intr_name="llvm.amdgcn.image.sample.c.d.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2666, intr_name="llvm.amdgcn.image.sample.c.d.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2667, intr_name="llvm.amdgcn.image.sample.c.d.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2668, intr_name="llvm.amdgcn.image.sample.c.d.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2669, intr_name="llvm.amdgcn.image.sample.c.d.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2670, intr_name="llvm.amdgcn.image.sample.c.d.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2671, intr_name="llvm.amdgcn.image.sample.c.d.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2672, intr_name="llvm.amdgcn.image.sample.c.d.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2673, intr_name="llvm.amdgcn.image.sample.c.d.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2674, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2675, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2676, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2677, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2678, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2679, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2680, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2681, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: anyfloat, k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: LLVMMatchType[Literal[2]], n: any, o: any, p: i1, q: i32, r: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, intr_id=2682, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: LLVMMatchType[Literal[0]], j: anyfloat, k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: LLVMMatchType[Literal[1]], n: any, o: any, p: i1, q: i32, r: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, intr_id=2683, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2684, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cl_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2685, intr_name="llvm.amdgcn.image.sample.c.d.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2686, intr_name="llvm.amdgcn.image.sample.c.d.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_d_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2687, intr_name="llvm.amdgcn.image.sample.c.d.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2688, intr_name="llvm.amdgcn.image.sample.c.d.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2689, intr_name="llvm.amdgcn.image.sample.c.d.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2690, intr_name="llvm.amdgcn.image.sample.c.d.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2691, intr_name="llvm.amdgcn.image.sample.c.d.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2692, intr_name="llvm.amdgcn.image.sample.c.d.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2693, intr_name="llvm.amdgcn.image.sample.c.d.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2694, intr_name="llvm.amdgcn.image.sample.c.d.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2695, intr_name="llvm.amdgcn.image.sample.c.d.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: anyfloat, k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2696, intr_name="llvm.amdgcn.image.sample.c.d.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: LLVMMatchType[Literal[0]], j: anyfloat, k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2697, intr_name="llvm.amdgcn.image.sample.c.d.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2698, intr_name="llvm.amdgcn.image.sample.c.d.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_d_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2699, intr_name="llvm.amdgcn.image.sample.c.d.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_1d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2700, intr_name="llvm.amdgcn.image.sample.c.l.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_1d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2701, intr_name="llvm.amdgcn.image.sample.c.l.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2702, intr_name="llvm.amdgcn.image.sample.c.l.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_l_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2703, intr_name="llvm.amdgcn.image.sample.c.l.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2704, intr_name="llvm.amdgcn.image.sample.c.l.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2705, intr_name="llvm.amdgcn.image.sample.c.l.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2706, intr_name="llvm.amdgcn.image.sample.c.l.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_l_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2707, intr_name="llvm.amdgcn.image.sample.c.l.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2708, intr_name="llvm.amdgcn.image.sample.c.l.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2709, intr_name="llvm.amdgcn.image.sample.c.l.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2710, intr_name="llvm.amdgcn.image.sample.c.l.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_l_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2711, intr_name="llvm.amdgcn.image.sample.c.l.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2712, intr_name="llvm.amdgcn.image.sample.c.l.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2713, intr_name="llvm.amdgcn.image.sample.c.l.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2714, intr_name="llvm.amdgcn.image.sample.c.l.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2715, intr_name="llvm.amdgcn.image.sample.c.l.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2716, intr_name="llvm.amdgcn.image.sample.c.l.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2717, intr_name="llvm.amdgcn.image.sample.c.l.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2718, intr_name="llvm.amdgcn.image.sample.c.l.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2719, intr_name="llvm.amdgcn.image.sample.c.l.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2720, intr_name="llvm.amdgcn.image.sample.c.l.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2721, intr_name="llvm.amdgcn.image.sample.c.l.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2722, intr_name="llvm.amdgcn.image.sample.c.l.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_l_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2723, intr_name="llvm.amdgcn.image.sample.c.l.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_1d(a: i32, b: float, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2724, intr_name="llvm.amdgcn.image.sample.c.lz.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_1d_nortn(a: i32, b: float, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2725, intr_name="llvm.amdgcn.image.sample.c.lz.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_1darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2726, intr_name="llvm.amdgcn.image.sample.c.lz.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_1darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2727, intr_name="llvm.amdgcn.image.sample.c.lz.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_2d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2728, intr_name="llvm.amdgcn.image.sample.c.lz.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_2d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2729, intr_name="llvm.amdgcn.image.sample.c.lz.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_2darray(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2730, intr_name="llvm.amdgcn.image.sample.c.lz.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_2darray_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2731, intr_name="llvm.amdgcn.image.sample.c.lz.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_3d(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2732, intr_name="llvm.amdgcn.image.sample.c.lz.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_3d_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2733, intr_name="llvm.amdgcn.image.sample.c.lz.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_cube(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2734, intr_name="llvm.amdgcn.image.sample.c.lz.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_cube_nortn(a: i32, b: float, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2735, intr_name="llvm.amdgcn.image.sample.c.lz.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2736, intr_name="llvm.amdgcn.image.sample.c.lz.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2737, intr_name="llvm.amdgcn.image.sample.c.lz.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2738, intr_name="llvm.amdgcn.image.sample.c.lz.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2739, intr_name="llvm.amdgcn.image.sample.c.lz.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2740, intr_name="llvm.amdgcn.image.sample.c.lz.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2741, intr_name="llvm.amdgcn.image.sample.c.lz.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2742, intr_name="llvm.amdgcn.image.sample.c.lz.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2743, intr_name="llvm.amdgcn.image.sample.c.lz.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2744, intr_name="llvm.amdgcn.image.sample.c.lz.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2745, intr_name="llvm.amdgcn.image.sample.c.lz.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2746, intr_name="llvm.amdgcn.image.sample.c.lz.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_lz_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2747, intr_name="llvm.amdgcn.image.sample.c.lz.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_1d(a: i32, b: i32, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2748, intr_name="llvm.amdgcn.image.sample.c.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_c_o_1d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2749, intr_name="llvm.amdgcn.image.sample.c.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_1darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2750, intr_name="llvm.amdgcn.image.sample.c.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_c_o_1darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2751, intr_name="llvm.amdgcn.image.sample.c.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_2d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2752, intr_name="llvm.amdgcn.image.sample.c.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_c_o_2d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2753, intr_name="llvm.amdgcn.image.sample.c.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_2darray(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2754, intr_name="llvm.amdgcn.image.sample.c.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_c_o_2darray_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2755, intr_name="llvm.amdgcn.image.sample.c.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_3d(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2756, intr_name="llvm.amdgcn.image.sample.c.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_c_o_3d_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2757, intr_name="llvm.amdgcn.image.sample.c.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_c_o_cube(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2758, intr_name="llvm.amdgcn.image.sample.c.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_c_o_cube_nortn(a: i32, b: i32, c: float, d: anyfloat, e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2759, intr_name="llvm.amdgcn.image.sample.c.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2760, intr_name="llvm.amdgcn.image.sample.cd.1d", is_overloaded=True, return_type=return_type)


def image_sample_cd_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2761, intr_name="llvm.amdgcn.image.sample.cd.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2762, intr_name="llvm.amdgcn.image.sample.cd.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2763, intr_name="llvm.amdgcn.image.sample.cd.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2764, intr_name="llvm.amdgcn.image.sample.cd.2d", is_overloaded=True, return_type=return_type)


def image_sample_cd_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2765, intr_name="llvm.amdgcn.image.sample.cd.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2766, intr_name="llvm.amdgcn.image.sample.cd.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2767, intr_name="llvm.amdgcn.image.sample.cd.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2768, intr_name="llvm.amdgcn.image.sample.cd.3d", is_overloaded=True, return_type=return_type)


def image_sample_cd_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2769, intr_name="llvm.amdgcn.image.sample.cd.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2770, intr_name="llvm.amdgcn.image.sample.cd.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2771, intr_name="llvm.amdgcn.image.sample.cd.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2772, intr_name="llvm.amdgcn.image.sample.cd.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2773, intr_name="llvm.amdgcn.image.sample.cd.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2774, intr_name="llvm.amdgcn.image.sample.cd.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2775, intr_name="llvm.amdgcn.image.sample.cd.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2776, intr_name="llvm.amdgcn.image.sample.cd.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2777, intr_name="llvm.amdgcn.image.sample.cd.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2778, intr_name="llvm.amdgcn.image.sample.cd.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2779, intr_name="llvm.amdgcn.image.sample.cd.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2780, intr_name="llvm.amdgcn.image.sample.cd.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2781, intr_name="llvm.amdgcn.image.sample.cd.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2782, intr_name="llvm.amdgcn.image.sample.cd.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2783, intr_name="llvm.amdgcn.image.sample.cd.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2784, intr_name="llvm.amdgcn.image.sample.cd.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2785, intr_name="llvm.amdgcn.image.sample.cd.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2786, intr_name="llvm.amdgcn.image.sample.cd.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2787, intr_name="llvm.amdgcn.image.sample.cd.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2788, intr_name="llvm.amdgcn.image.sample.cd.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2789, intr_name="llvm.amdgcn.image.sample.cd.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2790, intr_name="llvm.amdgcn.image.sample.cd.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2791, intr_name="llvm.amdgcn.image.sample.cd.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2792, intr_name="llvm.amdgcn.image.sample.cd.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_cd_cl_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2793, intr_name="llvm.amdgcn.image.sample.cd.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2794, intr_name="llvm.amdgcn.image.sample.cd.cube", is_overloaded=True, return_type=return_type)


def image_sample_cd_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2795, intr_name="llvm.amdgcn.image.sample.cd.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2796, intr_name="llvm.amdgcn.image.sample.cd.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2797, intr_name="llvm.amdgcn.image.sample.cd.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2798, intr_name="llvm.amdgcn.image.sample.cd.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2799, intr_name="llvm.amdgcn.image.sample.cd.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2800, intr_name="llvm.amdgcn.image.sample.cd.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2801, intr_name="llvm.amdgcn.image.sample.cd.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2802, intr_name="llvm.amdgcn.image.sample.cd.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2803, intr_name="llvm.amdgcn.image.sample.cd.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2804, intr_name="llvm.amdgcn.image.sample.cd.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2805, intr_name="llvm.amdgcn.image.sample.cd.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2806, intr_name="llvm.amdgcn.image.sample.cd.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_cd_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2807, intr_name="llvm.amdgcn.image.sample.cd.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2808, intr_name="llvm.amdgcn.image.sample.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_cl_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2809, intr_name="llvm.amdgcn.image.sample.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2810, intr_name="llvm.amdgcn.image.sample.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cl_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2811, intr_name="llvm.amdgcn.image.sample.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2812, intr_name="llvm.amdgcn.image.sample.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_cl_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2813, intr_name="llvm.amdgcn.image.sample.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2814, intr_name="llvm.amdgcn.image.sample.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cl_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2815, intr_name="llvm.amdgcn.image.sample.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2816, intr_name="llvm.amdgcn.image.sample.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_cl_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2817, intr_name="llvm.amdgcn.image.sample.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2818, intr_name="llvm.amdgcn.image.sample.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_cl_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2819, intr_name="llvm.amdgcn.image.sample.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2820, intr_name="llvm.amdgcn.image.sample.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2821, intr_name="llvm.amdgcn.image.sample.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2822, intr_name="llvm.amdgcn.image.sample.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2823, intr_name="llvm.amdgcn.image.sample.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2824, intr_name="llvm.amdgcn.image.sample.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2825, intr_name="llvm.amdgcn.image.sample.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2826, intr_name="llvm.amdgcn.image.sample.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2827, intr_name="llvm.amdgcn.image.sample.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2828, intr_name="llvm.amdgcn.image.sample.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2829, intr_name="llvm.amdgcn.image.sample.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2830, intr_name="llvm.amdgcn.image.sample.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_cl_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2831, intr_name="llvm.amdgcn.image.sample.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2832, intr_name="llvm.amdgcn.image.sample.cube", is_overloaded=True, return_type=return_type)


def image_sample_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2833, intr_name="llvm.amdgcn.image.sample.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2834, intr_name="llvm.amdgcn.image.sample.d.1d", is_overloaded=True, return_type=return_type)


def image_sample_d_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2835, intr_name="llvm.amdgcn.image.sample.d.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2836, intr_name="llvm.amdgcn.image.sample.d.1darray", is_overloaded=True, return_type=return_type)


def image_sample_d_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2837, intr_name="llvm.amdgcn.image.sample.d.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2838, intr_name="llvm.amdgcn.image.sample.d.2d", is_overloaded=True, return_type=return_type)


def image_sample_d_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2839, intr_name="llvm.amdgcn.image.sample.d.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2840, intr_name="llvm.amdgcn.image.sample.d.2darray", is_overloaded=True, return_type=return_type)


def image_sample_d_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2841, intr_name="llvm.amdgcn.image.sample.d.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2842, intr_name="llvm.amdgcn.image.sample.d.3d", is_overloaded=True, return_type=return_type)


def image_sample_d_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2843, intr_name="llvm.amdgcn.image.sample.d.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2844, intr_name="llvm.amdgcn.image.sample.d.cl.1d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2845, intr_name="llvm.amdgcn.image.sample.d.cl.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: anyfloat, e: LLVMMatchType[Literal[2]], f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2846, intr_name="llvm.amdgcn.image.sample.d.cl.1darray", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyfloat, e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2847, intr_name="llvm.amdgcn.image.sample.d.cl.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2848, intr_name="llvm.amdgcn.image.sample.d.cl.2d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2849, intr_name="llvm.amdgcn.image.sample.d.cl.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2850, intr_name="llvm.amdgcn.image.sample.d.cl.2darray", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2851, intr_name="llvm.amdgcn.image.sample.d.cl.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: anyfloat, i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2852, intr_name="llvm.amdgcn.image.sample.d.cl.3d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: anyfloat, i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2853, intr_name="llvm.amdgcn.image.sample.d.cl.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2854, intr_name="llvm.amdgcn.image.sample.d.cl.cube", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2855, intr_name="llvm.amdgcn.image.sample.d.cl.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2856, intr_name="llvm.amdgcn.image.sample.d.cl.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2857, intr_name="llvm.amdgcn.image.sample.d.cl.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: LLVMMatchType[Literal[2]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2858, intr_name="llvm.amdgcn.image.sample.d.cl.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: any, i: any, j: i1, k: i32, l: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, intr_id=2859, intr_name="llvm.amdgcn.image.sample.d.cl.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2860, intr_name="llvm.amdgcn.image.sample.d.cl.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2861, intr_name="llvm.amdgcn.image.sample.d.cl.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2862, intr_name="llvm.amdgcn.image.sample.d.cl.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2863, intr_name="llvm.amdgcn.image.sample.d.cl.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: LLVMMatchType[Literal[2]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2864, intr_name="llvm.amdgcn.image.sample.d.cl.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: LLVMMatchType[Literal[1]], m: any, n: any, o: i1, p: i32, q: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, intr_id=2865, intr_name="llvm.amdgcn.image.sample.d.cl.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: LLVMMatchType[Literal[2]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2866, intr_name="llvm.amdgcn.image.sample.d.cl.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_d_cl_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: LLVMMatchType[Literal[1]], k: any, l: any, m: i1, n: i32, o: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, intr_id=2867, intr_name="llvm.amdgcn.image.sample.d.cl.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: anyfloat, g: LLVMMatchType[Literal[2]], h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2868, intr_name="llvm.amdgcn.image.sample.d.cube", is_overloaded=True, return_type=return_type)


def image_sample_d_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: anyfloat, g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2869, intr_name="llvm.amdgcn.image.sample.d.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2870, intr_name="llvm.amdgcn.image.sample.d.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_d_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2871, intr_name="llvm.amdgcn.image.sample.d.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: anyfloat, f: LLVMMatchType[Literal[2]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2872, intr_name="llvm.amdgcn.image.sample.d.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_d_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: anyfloat, f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2873, intr_name="llvm.amdgcn.image.sample.d.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2874, intr_name="llvm.amdgcn.image.sample.d.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_d_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: any, j: any, k: i1, l: i32, m: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, intr_id=2875, intr_name="llvm.amdgcn.image.sample.d.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2876, intr_name="llvm.amdgcn.image.sample.d.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_d_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2877, intr_name="llvm.amdgcn.image.sample.d.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: LLVMMatchType[Literal[1]], h: LLVMMatchType[Literal[1]], i: anyfloat, j: LLVMMatchType[Literal[2]], k: LLVMMatchType[Literal[2]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2878, intr_name="llvm.amdgcn.image.sample.d.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_d_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: LLVMMatchType[Literal[0]], h: LLVMMatchType[Literal[0]], i: anyfloat, j: LLVMMatchType[Literal[1]], k: LLVMMatchType[Literal[1]], l: any, m: any, n: i1, o: i32, p: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, intr_id=2879, intr_name="llvm.amdgcn.image.sample.d.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_d_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: anyfloat, h: LLVMMatchType[Literal[2]], i: LLVMMatchType[Literal[2]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2880, intr_name="llvm.amdgcn.image.sample.d.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_d_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: anyfloat, h: LLVMMatchType[Literal[1]], i: LLVMMatchType[Literal[1]], j: any, k: any, l: i1, m: i32, n: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, l, m, n, intr_id=2881, intr_name="llvm.amdgcn.image.sample.d.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_1d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2882, intr_name="llvm.amdgcn.image.sample.l.1d", is_overloaded=True, return_type=return_type)


def image_sample_l_1d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2883, intr_name="llvm.amdgcn.image.sample.l.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2884, intr_name="llvm.amdgcn.image.sample.l.1darray", is_overloaded=True, return_type=return_type)


def image_sample_l_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2885, intr_name="llvm.amdgcn.image.sample.l.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2886, intr_name="llvm.amdgcn.image.sample.l.2d", is_overloaded=True, return_type=return_type)


def image_sample_l_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2887, intr_name="llvm.amdgcn.image.sample.l.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2888, intr_name="llvm.amdgcn.image.sample.l.2darray", is_overloaded=True, return_type=return_type)


def image_sample_l_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2889, intr_name="llvm.amdgcn.image.sample.l.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2890, intr_name="llvm.amdgcn.image.sample.l.3d", is_overloaded=True, return_type=return_type)


def image_sample_l_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2891, intr_name="llvm.amdgcn.image.sample.l.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2892, intr_name="llvm.amdgcn.image.sample.l.cube", is_overloaded=True, return_type=return_type)


def image_sample_l_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2893, intr_name="llvm.amdgcn.image.sample.l.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_1d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2894, intr_name="llvm.amdgcn.image.sample.l.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_l_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2895, intr_name="llvm.amdgcn.image.sample.l.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2896, intr_name="llvm.amdgcn.image.sample.l.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_l_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2897, intr_name="llvm.amdgcn.image.sample.l.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2898, intr_name="llvm.amdgcn.image.sample.l.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_l_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2899, intr_name="llvm.amdgcn.image.sample.l.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2900, intr_name="llvm.amdgcn.image.sample.l.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_l_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2901, intr_name="llvm.amdgcn.image.sample.l.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2902, intr_name="llvm.amdgcn.image.sample.l.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_l_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2903, intr_name="llvm.amdgcn.image.sample.l.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_l_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2904, intr_name="llvm.amdgcn.image.sample.l.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_l_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: LLVMMatchType[Literal[0]], g: any, h: any, i: i1, j: i32, k: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, k, intr_id=2905, intr_name="llvm.amdgcn.image.sample.l.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_1d(a: i32, b: anyfloat, c: any, d: any, e: i1, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=2906, intr_name="llvm.amdgcn.image.sample.lz.1d", is_overloaded=True, return_type=return_type)


def image_sample_lz_1d_nortn(a: i32, b: anyfloat, c: any, d: any, e: i1, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=2907, intr_name="llvm.amdgcn.image.sample.lz.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_1darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2908, intr_name="llvm.amdgcn.image.sample.lz.1darray", is_overloaded=True, return_type=return_type)


def image_sample_lz_1darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2909, intr_name="llvm.amdgcn.image.sample.lz.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_2d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2910, intr_name="llvm.amdgcn.image.sample.lz.2d", is_overloaded=True, return_type=return_type)


def image_sample_lz_2d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2911, intr_name="llvm.amdgcn.image.sample.lz.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_2darray(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2912, intr_name="llvm.amdgcn.image.sample.lz.2darray", is_overloaded=True, return_type=return_type)


def image_sample_lz_2darray_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2913, intr_name="llvm.amdgcn.image.sample.lz.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_3d(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2914, intr_name="llvm.amdgcn.image.sample.lz.3d", is_overloaded=True, return_type=return_type)


def image_sample_lz_3d_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2915, intr_name="llvm.amdgcn.image.sample.lz.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_cube(a: i32, b: anyfloat, c: LLVMMatchType[Literal[1]], d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2916, intr_name="llvm.amdgcn.image.sample.lz.cube", is_overloaded=True, return_type=return_type)


def image_sample_lz_cube_nortn(a: i32, b: anyfloat, c: LLVMMatchType[Literal[0]], d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2917, intr_name="llvm.amdgcn.image.sample.lz.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_1d(a: i32, b: i32, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2918, intr_name="llvm.amdgcn.image.sample.lz.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2919, intr_name="llvm.amdgcn.image.sample.lz.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2920, intr_name="llvm.amdgcn.image.sample.lz.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2921, intr_name="llvm.amdgcn.image.sample.lz.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2922, intr_name="llvm.amdgcn.image.sample.lz.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2923, intr_name="llvm.amdgcn.image.sample.lz.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2924, intr_name="llvm.amdgcn.image.sample.lz.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2925, intr_name="llvm.amdgcn.image.sample.lz.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2926, intr_name="llvm.amdgcn.image.sample.lz.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2927, intr_name="llvm.amdgcn.image.sample.lz.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2928, intr_name="llvm.amdgcn.image.sample.lz.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_lz_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2929, intr_name="llvm.amdgcn.image.sample.lz.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_1d(a: i32, b: i32, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2930, intr_name="llvm.amdgcn.image.sample.o.1d", is_overloaded=True, return_type=return_type)


def image_sample_o_1d_nortn(a: i32, b: i32, c: anyfloat, d: any, e: any, f: i1, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2931, intr_name="llvm.amdgcn.image.sample.o.1d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_1darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2932, intr_name="llvm.amdgcn.image.sample.o.1darray", is_overloaded=True, return_type=return_type)


def image_sample_o_1darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2933, intr_name="llvm.amdgcn.image.sample.o.1darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_2d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2934, intr_name="llvm.amdgcn.image.sample.o.2d", is_overloaded=True, return_type=return_type)


def image_sample_o_2d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: any, f: any, g: i1, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2935, intr_name="llvm.amdgcn.image.sample.o.2d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_2darray(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2936, intr_name="llvm.amdgcn.image.sample.o.2darray", is_overloaded=True, return_type=return_type)


def image_sample_o_2darray_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2937, intr_name="llvm.amdgcn.image.sample.o.2darray.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_3d(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2938, intr_name="llvm.amdgcn.image.sample.o.3d", is_overloaded=True, return_type=return_type)


def image_sample_o_3d_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2939, intr_name="llvm.amdgcn.image.sample.o.3d.nortn", is_overloaded=True, return_type=return_type)


def image_sample_o_cube(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2940, intr_name="llvm.amdgcn.image.sample.o.cube", is_overloaded=True, return_type=return_type)


def image_sample_o_cube_nortn(a: i32, b: i32, c: anyfloat, d: LLVMMatchType[Literal[0]], e: LLVMMatchType[Literal[0]], f: any, g: any, h: i1, i: i32, j: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, j, intr_id=2941, intr_name="llvm.amdgcn.image.sample.o.cube.nortn", is_overloaded=True, return_type=return_type)


def image_store_1d(a: anyfloat, b: i32, c: anyint, d: any, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=2942, intr_name="llvm.amdgcn.image.store.1d", is_overloaded=True, return_type=return_type)


def image_store_1darray(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=2943, intr_name="llvm.amdgcn.image.store.1darray", is_overloaded=True, return_type=return_type)


def image_store_2d(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=2944, intr_name="llvm.amdgcn.image.store.2d", is_overloaded=True, return_type=return_type)


def image_store_2darray(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2945, intr_name="llvm.amdgcn.image.store.2darray", is_overloaded=True, return_type=return_type)


def image_store_2darraymsaa(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2946, intr_name="llvm.amdgcn.image.store.2darraymsaa", is_overloaded=True, return_type=return_type)


def image_store_2dmsaa(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2947, intr_name="llvm.amdgcn.image.store.2dmsaa", is_overloaded=True, return_type=return_type)


def image_store_3d(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2948, intr_name="llvm.amdgcn.image.store.3d", is_overloaded=True, return_type=return_type)


def image_store_cube(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2949, intr_name="llvm.amdgcn.image.store.cube", is_overloaded=True, return_type=return_type)


def image_store_mip_1d(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: any, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=2950, intr_name="llvm.amdgcn.image.store.mip.1d", is_overloaded=True, return_type=return_type)


def image_store_mip_1darray(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2951, intr_name="llvm.amdgcn.image.store.mip.1darray", is_overloaded=True, return_type=return_type)


def image_store_mip_2d(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: any, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=2952, intr_name="llvm.amdgcn.image.store.mip.2d", is_overloaded=True, return_type=return_type)


def image_store_mip_2darray(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2953, intr_name="llvm.amdgcn.image.store.mip.2darray", is_overloaded=True, return_type=return_type)


def image_store_mip_3d(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2954, intr_name="llvm.amdgcn.image.store.mip.3d", is_overloaded=True, return_type=return_type)


def image_store_mip_cube(a: anyfloat, b: i32, c: anyint, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[1]], f: LLVMMatchType[Literal[1]], g: any, h: i32, i: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=2955, intr_name="llvm.amdgcn.image.store.mip.cube", is_overloaded=True, return_type=return_type)


def implicit_buffer_ptr(return_type=None):
    return call_intrinsic(intr_id=2956, intr_name="llvm.amdgcn.implicit.buffer.ptr", is_overloaded=False, return_type=return_type)


def implicitarg_ptr(return_type=None):
    return call_intrinsic(intr_id=2957, intr_name="llvm.amdgcn.implicitarg.ptr", is_overloaded=False, return_type=return_type)


def init_exec(a: i64, return_type=None):
    call_intrinsic(a, intr_id=2958, intr_name="llvm.amdgcn.init.exec", is_overloaded=False, return_type=return_type)


def init_exec_from_input(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=2959, intr_name="llvm.amdgcn.init.exec.from.input", is_overloaded=False, return_type=return_type)


def init_whole_wave(return_type=None):
    return call_intrinsic(intr_id=2960, intr_name="llvm.amdgcn.init.whole.wave", is_overloaded=False, return_type=return_type)


def interp_inreg_p10(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2961, intr_name="llvm.amdgcn.interp.inreg.p10", is_overloaded=False, return_type=return_type)


def interp_inreg_p10_f16(a: float, b: float, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2962, intr_name="llvm.amdgcn.interp.inreg.p10.f16", is_overloaded=False, return_type=return_type)


def interp_inreg_p2(a: float, b: float, c: float, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2963, intr_name="llvm.amdgcn.interp.inreg.p2", is_overloaded=False, return_type=return_type)


def interp_inreg_p2_f16(a: float, b: float, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2964, intr_name="llvm.amdgcn.interp.inreg.p2.f16", is_overloaded=False, return_type=return_type)


def interp_mov(a: i32, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2965, intr_name="llvm.amdgcn.interp.mov", is_overloaded=False, return_type=return_type)


def interp_p1(a: float, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2966, intr_name="llvm.amdgcn.interp.p1", is_overloaded=False, return_type=return_type)


def interp_p10_rtz_f16(a: float, b: float, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2968, intr_name="llvm.amdgcn.interp.p10.rtz.f16", is_overloaded=False, return_type=return_type)


def interp_p1_f16(a: float, b: i32, c: i32, d: i1, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2967, intr_name="llvm.amdgcn.interp.p1.f16", is_overloaded=False, return_type=return_type)


def interp_p2(a: float, b: float, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=2969, intr_name="llvm.amdgcn.interp.p2", is_overloaded=False, return_type=return_type)


def interp_p2_f16(a: float, b: float, c: i32, d: i32, e: i1, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2970, intr_name="llvm.amdgcn.interp.p2.f16", is_overloaded=False, return_type=return_type)


def interp_p2_rtz_f16(a: float, b: float, c: float, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2971, intr_name="llvm.amdgcn.interp.p2.rtz.f16", is_overloaded=False, return_type=return_type)


def inverse_ballot(a: anyint, return_type=None):
    return call_intrinsic(a, intr_id=2972, intr_name="llvm.amdgcn.inverse.ballot", is_overloaded=True, return_type=return_type)


def is_private(a: pointer, return_type=None):
    return call_intrinsic(a, intr_id=2973, intr_name="llvm.amdgcn.is.private", is_overloaded=False, return_type=return_type)


def is_shared(a: pointer, return_type=None):
    return call_intrinsic(a, intr_id=2974, intr_name="llvm.amdgcn.is.shared", is_overloaded=False, return_type=return_type)


def kernarg_segment_ptr(return_type=None):
    return call_intrinsic(intr_id=2975, intr_name="llvm.amdgcn.kernarg.segment.ptr", is_overloaded=False, return_type=return_type)


def kill(a: i1, return_type=None):
    call_intrinsic(a, intr_id=2976, intr_name="llvm.amdgcn.kill", is_overloaded=False, return_type=return_type)


def lds_direct_load(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=2977, intr_name="llvm.amdgcn.lds.direct.load", is_overloaded=True, return_type=return_type)


def lds_kernel_id(return_type=None):
    return call_intrinsic(intr_id=2978, intr_name="llvm.amdgcn.lds.kernel.id", is_overloaded=False, return_type=return_type)


def lds_param_load(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2979, intr_name="llvm.amdgcn.lds.param.load", is_overloaded=False, return_type=return_type)


def lerp(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=2980, intr_name="llvm.amdgcn.lerp", is_overloaded=False, return_type=return_type)


def live_mask(return_type=None):
    return call_intrinsic(intr_id=2981, intr_name="llvm.amdgcn.live.mask", is_overloaded=False, return_type=return_type)


def log(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=2982, intr_name="llvm.amdgcn.log", is_overloaded=True, return_type=return_type)


def log_clamp(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=2983, intr_name="llvm.amdgcn.log.clamp", is_overloaded=True, return_type=return_type)


def loop(a: anyint, return_type=None):
    return call_intrinsic(a, intr_id=2984, intr_name="llvm.amdgcn.loop", is_overloaded=True, return_type=return_type)


def make_buffer_rsrc(a: anyptr, b: i16, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=2985, intr_name="llvm.amdgcn.make.buffer.rsrc", is_overloaded=True, return_type=return_type)


def mbcnt_hi(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2986, intr_name="llvm.amdgcn.mbcnt.hi", is_overloaded=False, return_type=return_type)


def mbcnt_lo(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=2987, intr_name="llvm.amdgcn.mbcnt.lo", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x16bf16_1k(a: v4i16, b: v4i16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2988, intr_name="llvm.amdgcn.mfma.f32.16x16x16bf16.1k", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x16f16(a: v4f16, b: v4f16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2989, intr_name="llvm.amdgcn.mfma.f32.16x16x16f16", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x1f32(a: float, b: float, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2990, intr_name="llvm.amdgcn.mfma.f32.16x16x1f32", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x2bf16(a: v2i16, b: v2i16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2991, intr_name="llvm.amdgcn.mfma.f32.16x16x2bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_bf16(a: v8bf16, b: v8bf16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2992, intr_name="llvm.amdgcn.mfma.f32.16x16x32.bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_bf8_bf8(a: i64, b: i64, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2993, intr_name="llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_bf8_fp8(a: i64, b: i64, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2994, intr_name="llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_f16(a: v8f16, b: v8f16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2995, intr_name="llvm.amdgcn.mfma.f32.16x16x32.f16", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_fp8_bf8(a: i64, b: i64, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2996, intr_name="llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x32_fp8_fp8(a: i64, b: i64, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2997, intr_name="llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x4bf16_1k(a: v4i16, b: v4i16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2998, intr_name="llvm.amdgcn.mfma.f32.16x16x4bf16.1k", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x4f16(a: v4f16, b: v4f16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=2999, intr_name="llvm.amdgcn.mfma.f32.16x16x4f16", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x4f32(a: float, b: float, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3000, intr_name="llvm.amdgcn.mfma.f32.16x16x4f32", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x8_xf32(a: v2f32, b: v2f32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3001, intr_name="llvm.amdgcn.mfma.f32.16x16x8.xf32", is_overloaded=False, return_type=return_type)


def mfma_f32_16x16x8bf16(a: v2i16, b: v2i16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3002, intr_name="llvm.amdgcn.mfma.f32.16x16x8bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_bf16(a: v8bf16, b: v8bf16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3003, intr_name="llvm.amdgcn.mfma.f32.32x32x16.bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_bf8_bf8(a: i64, b: i64, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3004, intr_name="llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_bf8_fp8(a: i64, b: i64, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3005, intr_name="llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_f16(a: v8f16, b: v8f16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3006, intr_name="llvm.amdgcn.mfma.f32.32x32x16.f16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_fp8_bf8(a: i64, b: i64, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3007, intr_name="llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x16_fp8_fp8(a: i64, b: i64, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3008, intr_name="llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x1f32(a: float, b: float, c: v32f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3009, intr_name="llvm.amdgcn.mfma.f32.32x32x1f32", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x2bf16(a: v2i16, b: v2i16, c: v32f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3010, intr_name="llvm.amdgcn.mfma.f32.32x32x2bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x2f32(a: float, b: float, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3011, intr_name="llvm.amdgcn.mfma.f32.32x32x2f32", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x4_xf32(a: v2f32, b: v2f32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3012, intr_name="llvm.amdgcn.mfma.f32.32x32x4.xf32", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x4bf16(a: v2i16, b: v2i16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3013, intr_name="llvm.amdgcn.mfma.f32.32x32x4bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x4bf16_1k(a: v4i16, b: v4i16, c: v32f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3014, intr_name="llvm.amdgcn.mfma.f32.32x32x4bf16.1k", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x4f16(a: v4f16, b: v4f16, c: v32f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3015, intr_name="llvm.amdgcn.mfma.f32.32x32x4f16", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x8bf16_1k(a: v4i16, b: v4i16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3016, intr_name="llvm.amdgcn.mfma.f32.32x32x8bf16.1k", is_overloaded=False, return_type=return_type)


def mfma_f32_32x32x8f16(a: v4f16, b: v4f16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3017, intr_name="llvm.amdgcn.mfma.f32.32x32x8f16", is_overloaded=False, return_type=return_type)


def mfma_f32_4x4x1f32(a: float, b: float, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3018, intr_name="llvm.amdgcn.mfma.f32.4x4x1f32", is_overloaded=False, return_type=return_type)


def mfma_f32_4x4x2bf16(a: v2i16, b: v2i16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3019, intr_name="llvm.amdgcn.mfma.f32.4x4x2bf16", is_overloaded=False, return_type=return_type)


def mfma_f32_4x4x4bf16_1k(a: v4i16, b: v4i16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3020, intr_name="llvm.amdgcn.mfma.f32.4x4x4bf16.1k", is_overloaded=False, return_type=return_type)


def mfma_f32_4x4x4f16(a: v4f16, b: v4f16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3021, intr_name="llvm.amdgcn.mfma.f32.4x4x4f16", is_overloaded=False, return_type=return_type)


def mfma_f64_16x16x4f64(a: f64, b: f64, c: v4f64, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3022, intr_name="llvm.amdgcn.mfma.f64.16x16x4f64", is_overloaded=False, return_type=return_type)


def mfma_f64_4x4x4f64(a: f64, b: f64, c: f64, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3023, intr_name="llvm.amdgcn.mfma.f64.4x4x4f64", is_overloaded=False, return_type=return_type)


def mfma_i32_16x16x16i8(a: i32, b: i32, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3024, intr_name="llvm.amdgcn.mfma.i32.16x16x16i8", is_overloaded=False, return_type=return_type)


def mfma_i32_16x16x32_i8(a: i64, b: i64, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3025, intr_name="llvm.amdgcn.mfma.i32.16x16x32.i8", is_overloaded=False, return_type=return_type)


def mfma_i32_16x16x4i8(a: i32, b: i32, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3026, intr_name="llvm.amdgcn.mfma.i32.16x16x4i8", is_overloaded=False, return_type=return_type)


def mfma_i32_16x16x64_i8(a: v4i32, b: v4i32, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3027, intr_name="llvm.amdgcn.mfma.i32.16x16x64.i8", is_overloaded=False, return_type=return_type)


def mfma_i32_32x32x16_i8(a: i64, b: i64, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3028, intr_name="llvm.amdgcn.mfma.i32.32x32x16.i8", is_overloaded=False, return_type=return_type)


def mfma_i32_32x32x32_i8(a: v4i32, b: v4i32, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3029, intr_name="llvm.amdgcn.mfma.i32.32x32x32.i8", is_overloaded=False, return_type=return_type)


def mfma_i32_32x32x4i8(a: i32, b: i32, c: v32i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3030, intr_name="llvm.amdgcn.mfma.i32.32x32x4i8", is_overloaded=False, return_type=return_type)


def mfma_i32_32x32x8i8(a: i32, b: i32, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3031, intr_name="llvm.amdgcn.mfma.i32.32x32x8i8", is_overloaded=False, return_type=return_type)


def mfma_i32_4x4x4i8(a: i32, b: i32, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3032, intr_name="llvm.amdgcn.mfma.i32.4x4x4i8", is_overloaded=False, return_type=return_type)


def mfma_scale_f32_16x16x128_f8f6f4(a: anyvector, b: anyvector, c: v4f32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=3033, intr_name="llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", is_overloaded=True, return_type=return_type)


def mfma_scale_f32_32x32x64_f8f6f4(a: anyvector, b: anyvector, c: v16f32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, h, i, intr_id=3034, intr_name="llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4", is_overloaded=True, return_type=return_type)


def mov_dpp(a: LLVMMatchType[Literal[0]], b: i32, c: i32, d: i32, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3035, intr_name="llvm.amdgcn.mov.dpp", is_overloaded=True, return_type=return_type)


def mov_dpp8(a: LLVMMatchType[Literal[0]], b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3036, intr_name="llvm.amdgcn.mov.dpp8", is_overloaded=True, return_type=return_type)


def mqsad_pk_u16_u8(a: i64, b: i32, c: i64, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3037, intr_name="llvm.amdgcn.mqsad.pk.u16.u8", is_overloaded=False, return_type=return_type)


def mqsad_u32_u8(a: i64, b: i32, c: v4i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3038, intr_name="llvm.amdgcn.mqsad.u32.u8", is_overloaded=False, return_type=return_type)


def msad_u8(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3039, intr_name="llvm.amdgcn.msad.u8", is_overloaded=False, return_type=return_type)


def mul_i24(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3040, intr_name="llvm.amdgcn.mul.i24", is_overloaded=True, return_type=return_type)


def mul_u24(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3041, intr_name="llvm.amdgcn.mul.u24", is_overloaded=True, return_type=return_type)


def mulhi_i24(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3042, intr_name="llvm.amdgcn.mulhi.i24", is_overloaded=False, return_type=return_type)


def mulhi_u24(a: i32, b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3043, intr_name="llvm.amdgcn.mulhi.u24", is_overloaded=False, return_type=return_type)


def perm(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3044, intr_name="llvm.amdgcn.perm", is_overloaded=False, return_type=return_type)


def permlane16(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: i32, d: i32, e: i1, f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3045, intr_name="llvm.amdgcn.permlane16", is_overloaded=True, return_type=return_type)


def permlane16_swap(a: i32, b: i32, c: i1, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3046, intr_name="llvm.amdgcn.permlane16.swap", is_overloaded=False, return_type=return_type)


def permlane16_var(a: i32, b: i32, c: i32, d: i1, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3047, intr_name="llvm.amdgcn.permlane16.var", is_overloaded=False, return_type=return_type)


def permlane32_swap(a: i32, b: i32, c: i1, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3048, intr_name="llvm.amdgcn.permlane32.swap", is_overloaded=False, return_type=return_type)


def permlane64(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3049, intr_name="llvm.amdgcn.permlane64", is_overloaded=True, return_type=return_type)


def permlanex16(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: i32, d: i32, e: i1, f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3050, intr_name="llvm.amdgcn.permlanex16", is_overloaded=True, return_type=return_type)


def permlanex16_var(a: i32, b: i32, c: i32, d: i1, e: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3051, intr_name="llvm.amdgcn.permlanex16.var", is_overloaded=False, return_type=return_type)


def pops_exiting_wave_id(return_type=None):
    return call_intrinsic(intr_id=3052, intr_name="llvm.amdgcn.pops.exiting.wave.id", is_overloaded=False, return_type=return_type)


def prng_b32(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3053, intr_name="llvm.amdgcn.prng.b32", is_overloaded=False, return_type=return_type)


def ps_live(return_type=None):
    return call_intrinsic(intr_id=3054, intr_name="llvm.amdgcn.ps.live", is_overloaded=False, return_type=return_type)


def qsad_pk_u16_u8(a: i64, b: i32, c: i64, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3055, intr_name="llvm.amdgcn.qsad.pk.u16.u8", is_overloaded=False, return_type=return_type)


def queue_ptr(return_type=None):
    return call_intrinsic(intr_id=3056, intr_name="llvm.amdgcn.queue.ptr", is_overloaded=False, return_type=return_type)


def raw_atomic_buffer_load(a: v4i32, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3057, intr_name="llvm.amdgcn.raw.atomic.buffer.load", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_add(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3058, intr_name="llvm.amdgcn.raw.buffer.atomic.add", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_and(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3059, intr_name="llvm.amdgcn.raw.buffer.atomic.and", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_cmpswap(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3060, intr_name="llvm.amdgcn.raw.buffer.atomic.cmpswap", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_cond_sub_u32(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3061, intr_name="llvm.amdgcn.raw.buffer.atomic.cond.sub.u32", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_dec(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3062, intr_name="llvm.amdgcn.raw.buffer.atomic.dec", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_fadd(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3063, intr_name="llvm.amdgcn.raw.buffer.atomic.fadd", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_fmax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3064, intr_name="llvm.amdgcn.raw.buffer.atomic.fmax", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_fmin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3065, intr_name="llvm.amdgcn.raw.buffer.atomic.fmin", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_inc(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3066, intr_name="llvm.amdgcn.raw.buffer.atomic.inc", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_or(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3067, intr_name="llvm.amdgcn.raw.buffer.atomic.or", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_smax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3068, intr_name="llvm.amdgcn.raw.buffer.atomic.smax", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_smin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3069, intr_name="llvm.amdgcn.raw.buffer.atomic.smin", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_sub(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3070, intr_name="llvm.amdgcn.raw.buffer.atomic.sub", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_swap(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3071, intr_name="llvm.amdgcn.raw.buffer.atomic.swap", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_umax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3072, intr_name="llvm.amdgcn.raw.buffer.atomic.umax", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_umin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3073, intr_name="llvm.amdgcn.raw.buffer.atomic.umin", is_overloaded=True, return_type=return_type)


def raw_buffer_atomic_xor(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3074, intr_name="llvm.amdgcn.raw.buffer.atomic.xor", is_overloaded=True, return_type=return_type)


def raw_buffer_load(a: v4i32, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3075, intr_name="llvm.amdgcn.raw.buffer.load", is_overloaded=True, return_type=return_type)


def raw_buffer_load_format(a: v4i32, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3076, intr_name="llvm.amdgcn.raw.buffer.load.format", is_overloaded=True, return_type=return_type)


def raw_buffer_load_lds(a: v4i32, b: LLVMQualPointerType[Literal[3]], c: i32, d: i32, e: i32, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=3077, intr_name="llvm.amdgcn.raw.buffer.load.lds", is_overloaded=False, return_type=return_type)


def raw_buffer_store(a: any, b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, intr_id=3078, intr_name="llvm.amdgcn.raw.buffer.store", is_overloaded=True, return_type=return_type)


def raw_buffer_store_format(a: anyfloat, b: v4i32, c: i32, d: i32, e: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, intr_id=3079, intr_name="llvm.amdgcn.raw.buffer.store.format", is_overloaded=True, return_type=return_type)


def raw_ptr_atomic_buffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3080, intr_name="llvm.amdgcn.raw.ptr.atomic.buffer.load", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_add(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3081, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.add", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_and(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3082, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.and", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_cmpswap(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: AMDGPUBufferRsrcTy, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3083, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_cond_sub_u32(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3084, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.cond.sub.u32", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_dec(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3085, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.dec", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_fadd(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3086, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.fadd", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_fmax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3087, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.fmax", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_fmin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3088, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.fmin", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_inc(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3089, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.inc", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_or(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3090, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.or", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_smax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3091, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.smax", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_smin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3092, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.smin", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_sub(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3093, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.sub", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_swap(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3094, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.swap", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_umax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3095, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.umax", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_umin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3096, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.umin", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_atomic_xor(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3097, intr_name="llvm.amdgcn.raw.ptr.buffer.atomic.xor", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3098, intr_name="llvm.amdgcn.raw.ptr.buffer.load", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_load_format(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3099, intr_name="llvm.amdgcn.raw.ptr.buffer.load.format", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_load_lds(a: AMDGPUBufferRsrcTy, b: LLVMQualPointerType[Literal[3]], c: i32, d: i32, e: i32, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=3100, intr_name="llvm.amdgcn.raw.ptr.buffer.load.lds", is_overloaded=False, return_type=return_type)


def raw_ptr_buffer_store(a: any, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, intr_id=3101, intr_name="llvm.amdgcn.raw.ptr.buffer.store", is_overloaded=True, return_type=return_type)


def raw_ptr_buffer_store_format(a: anyfloat, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, intr_id=3102, intr_name="llvm.amdgcn.raw.ptr.buffer.store.format", is_overloaded=True, return_type=return_type)


def raw_ptr_tbuffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3103, intr_name="llvm.amdgcn.raw.ptr.tbuffer.load", is_overloaded=True, return_type=return_type)


def raw_ptr_tbuffer_store(a: any, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3104, intr_name="llvm.amdgcn.raw.ptr.tbuffer.store", is_overloaded=True, return_type=return_type)


def raw_tbuffer_load(a: v4i32, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3105, intr_name="llvm.amdgcn.raw.tbuffer.load", is_overloaded=True, return_type=return_type)


def raw_tbuffer_store(a: any, b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3106, intr_name="llvm.amdgcn.raw.tbuffer.store", is_overloaded=True, return_type=return_type)


def rcp(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3107, intr_name="llvm.amdgcn.rcp", is_overloaded=True, return_type=return_type)


def rcp_legacy(a: float, return_type=None):
    return call_intrinsic(a, intr_id=3108, intr_name="llvm.amdgcn.rcp.legacy", is_overloaded=False, return_type=return_type)


def readfirstlane(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3109, intr_name="llvm.amdgcn.readfirstlane", is_overloaded=True, return_type=return_type)


def readlane(a: LLVMMatchType[Literal[0]], b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3110, intr_name="llvm.amdgcn.readlane", is_overloaded=True, return_type=return_type)


def reloc_constant(a: metadata, return_type=None):
    return call_intrinsic(a, intr_id=3111, intr_name="llvm.amdgcn.reloc.constant", is_overloaded=False, return_type=return_type)


def rsq(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3112, intr_name="llvm.amdgcn.rsq", is_overloaded=True, return_type=return_type)


def rsq_clamp(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3113, intr_name="llvm.amdgcn.rsq.clamp", is_overloaded=True, return_type=return_type)


def rsq_legacy(a: float, return_type=None):
    return call_intrinsic(a, intr_id=3114, intr_name="llvm.amdgcn.rsq.legacy", is_overloaded=False, return_type=return_type)


def s_barrier(return_type=None):
    call_intrinsic(intr_id=3115, intr_name="llvm.amdgcn.s.barrier", is_overloaded=False, return_type=return_type)


def s_barrier_init(a: local_ptr, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3116, intr_name="llvm.amdgcn.s.barrier.init", is_overloaded=False, return_type=return_type)


def s_barrier_join(a: local_ptr, return_type=None):
    call_intrinsic(a, intr_id=3117, intr_name="llvm.amdgcn.s.barrier.join", is_overloaded=False, return_type=return_type)


def s_barrier_leave(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3118, intr_name="llvm.amdgcn.s.barrier.leave", is_overloaded=False, return_type=return_type)


def s_barrier_signal(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3119, intr_name="llvm.amdgcn.s.barrier.signal", is_overloaded=False, return_type=return_type)


def s_barrier_signal_isfirst(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3120, intr_name="llvm.amdgcn.s.barrier.signal.isfirst", is_overloaded=False, return_type=return_type)


def s_barrier_signal_var(a: local_ptr, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3121, intr_name="llvm.amdgcn.s.barrier.signal.var", is_overloaded=False, return_type=return_type)


def s_barrier_wait(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3122, intr_name="llvm.amdgcn.s.barrier.wait", is_overloaded=False, return_type=return_type)


def s_bitreplicate(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3123, intr_name="llvm.amdgcn.s.bitreplicate", is_overloaded=False, return_type=return_type)


def s_buffer_load(a: v4i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3124, intr_name="llvm.amdgcn.s.buffer.load", is_overloaded=True, return_type=return_type)


def s_buffer_prefetch_data(a: AMDGPUBufferRsrcTy, b: i32, c: i32, return_type=None):
    call_intrinsic(a, b, c, intr_id=3125, intr_name="llvm.amdgcn.s.buffer.prefetch.data", is_overloaded=False, return_type=return_type)


def s_dcache_inv(return_type=None):
    call_intrinsic(intr_id=3126, intr_name="llvm.amdgcn.s.dcache.inv", is_overloaded=False, return_type=return_type)


def s_dcache_inv_vol(return_type=None):
    call_intrinsic(intr_id=3127, intr_name="llvm.amdgcn.s.dcache.inv.vol", is_overloaded=False, return_type=return_type)


def s_dcache_wb(return_type=None):
    call_intrinsic(intr_id=3128, intr_name="llvm.amdgcn.s.dcache.wb", is_overloaded=False, return_type=return_type)


def s_dcache_wb_vol(return_type=None):
    call_intrinsic(intr_id=3129, intr_name="llvm.amdgcn.s.dcache.wb.vol", is_overloaded=False, return_type=return_type)


def s_decperflevel(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3130, intr_name="llvm.amdgcn.s.decperflevel", is_overloaded=False, return_type=return_type)


def s_get_barrier_state(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3131, intr_name="llvm.amdgcn.s.get.barrier.state", is_overloaded=False, return_type=return_type)


def s_get_named_barrier_state(a: local_ptr, return_type=None):
    return call_intrinsic(a, intr_id=3132, intr_name="llvm.amdgcn.s.get.named.barrier.state", is_overloaded=False, return_type=return_type)


def s_get_waveid_in_workgroup(return_type=None):
    return call_intrinsic(intr_id=3133, intr_name="llvm.amdgcn.s.get.waveid.in.workgroup", is_overloaded=False, return_type=return_type)


def s_getpc(return_type=None):
    return call_intrinsic(intr_id=3134, intr_name="llvm.amdgcn.s.getpc", is_overloaded=False, return_type=return_type)


def s_getreg(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3135, intr_name="llvm.amdgcn.s.getreg", is_overloaded=False, return_type=return_type)


def s_incperflevel(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3136, intr_name="llvm.amdgcn.s.incperflevel", is_overloaded=False, return_type=return_type)


def s_memrealtime(return_type=None):
    return call_intrinsic(intr_id=3137, intr_name="llvm.amdgcn.s.memrealtime", is_overloaded=False, return_type=return_type)


def s_memtime(return_type=None):
    return call_intrinsic(intr_id=3138, intr_name="llvm.amdgcn.s.memtime", is_overloaded=False, return_type=return_type)


def s_nop(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3139, intr_name="llvm.amdgcn.s.nop", is_overloaded=False, return_type=return_type)


def s_prefetch_data(a: anyptr, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3140, intr_name="llvm.amdgcn.s.prefetch.data", is_overloaded=True, return_type=return_type)


def s_quadmask(a: anyint, return_type=None):
    return call_intrinsic(a, intr_id=3141, intr_name="llvm.amdgcn.s.quadmask", is_overloaded=True, return_type=return_type)


def s_sendmsg(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3142, intr_name="llvm.amdgcn.s.sendmsg", is_overloaded=False, return_type=return_type)


def s_sendmsg_rtn(a: i32, return_type=None):
    return call_intrinsic(a, intr_id=3143, intr_name="llvm.amdgcn.s.sendmsg.rtn", is_overloaded=True, return_type=return_type)


def s_sendmsghalt(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3144, intr_name="llvm.amdgcn.s.sendmsghalt", is_overloaded=False, return_type=return_type)


def s_sethalt(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3145, intr_name="llvm.amdgcn.s.sethalt", is_overloaded=False, return_type=return_type)


def s_setprio(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3146, intr_name="llvm.amdgcn.s.setprio", is_overloaded=False, return_type=return_type)


def s_setreg(a: i32, b: i32, return_type=None):
    call_intrinsic(a, b, intr_id=3147, intr_name="llvm.amdgcn.s.setreg", is_overloaded=False, return_type=return_type)


def s_sleep(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3148, intr_name="llvm.amdgcn.s.sleep", is_overloaded=False, return_type=return_type)


def s_sleep_var(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3149, intr_name="llvm.amdgcn.s.sleep.var", is_overloaded=False, return_type=return_type)


def s_ttracedata(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3150, intr_name="llvm.amdgcn.s.ttracedata", is_overloaded=False, return_type=return_type)


def s_ttracedata_imm(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3151, intr_name="llvm.amdgcn.s.ttracedata.imm", is_overloaded=False, return_type=return_type)


def s_wait_bvhcnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3152, intr_name="llvm.amdgcn.s.wait.bvhcnt", is_overloaded=False, return_type=return_type)


def s_wait_dscnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3153, intr_name="llvm.amdgcn.s.wait.dscnt", is_overloaded=False, return_type=return_type)


def s_wait_event_export_ready(return_type=None):
    call_intrinsic(intr_id=3154, intr_name="llvm.amdgcn.s.wait.event.export.ready", is_overloaded=False, return_type=return_type)


def s_wait_expcnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3155, intr_name="llvm.amdgcn.s.wait.expcnt", is_overloaded=False, return_type=return_type)


def s_wait_kmcnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3156, intr_name="llvm.amdgcn.s.wait.kmcnt", is_overloaded=False, return_type=return_type)


def s_wait_loadcnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3157, intr_name="llvm.amdgcn.s.wait.loadcnt", is_overloaded=False, return_type=return_type)


def s_wait_samplecnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3158, intr_name="llvm.amdgcn.s.wait.samplecnt", is_overloaded=False, return_type=return_type)


def s_wait_storecnt(a: i16, return_type=None):
    call_intrinsic(a, intr_id=3159, intr_name="llvm.amdgcn.s.wait.storecnt", is_overloaded=False, return_type=return_type)


def s_waitcnt(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3160, intr_name="llvm.amdgcn.s.waitcnt", is_overloaded=False, return_type=return_type)


def s_wqm(a: anyint, return_type=None):
    return call_intrinsic(a, intr_id=3161, intr_name="llvm.amdgcn.s.wqm", is_overloaded=True, return_type=return_type)


def sad_hi_u8(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3162, intr_name="llvm.amdgcn.sad.hi.u8", is_overloaded=False, return_type=return_type)


def sad_u16(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3163, intr_name="llvm.amdgcn.sad.u16", is_overloaded=False, return_type=return_type)


def sad_u8(a: i32, b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3164, intr_name="llvm.amdgcn.sad.u8", is_overloaded=False, return_type=return_type)


def sbfe(a: LLVMMatchType[Literal[0]], b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3165, intr_name="llvm.amdgcn.sbfe", is_overloaded=True, return_type=return_type)


def sched_barrier(a: i32, return_type=None):
    call_intrinsic(a, intr_id=3166, intr_name="llvm.amdgcn.sched.barrier", is_overloaded=False, return_type=return_type)


def sched_group_barrier(a: i32, b: i32, c: i32, return_type=None):
    call_intrinsic(a, b, c, intr_id=3167, intr_name="llvm.amdgcn.sched.group.barrier", is_overloaded=False, return_type=return_type)


def sdot2(a: v2i16, b: v2i16, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3168, intr_name="llvm.amdgcn.sdot2", is_overloaded=False, return_type=return_type)


def sdot4(a: i32, b: i32, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3169, intr_name="llvm.amdgcn.sdot4", is_overloaded=False, return_type=return_type)


def sdot8(a: i32, b: i32, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3170, intr_name="llvm.amdgcn.sdot8", is_overloaded=False, return_type=return_type)


def set_inactive(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, intr_id=3171, intr_name="llvm.amdgcn.set.inactive", is_overloaded=True, return_type=return_type)


def set_inactive_chain_arg(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, intr_id=3172, intr_name="llvm.amdgcn.set.inactive.chain.arg", is_overloaded=True, return_type=return_type)


def sffbh(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3173, intr_name="llvm.amdgcn.sffbh", is_overloaded=True, return_type=return_type)


def sin(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3174, intr_name="llvm.amdgcn.sin", is_overloaded=True, return_type=return_type)


def smfmac_f32_16x16x128_bf8_bf8(a: v4i32, b: v8i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3175, intr_name="llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x128_bf8_fp8(a: v4i32, b: v8i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3176, intr_name="llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x128_fp8_bf8(a: v4i32, b: v8i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3177, intr_name="llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x128_fp8_fp8(a: v4i32, b: v8i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3178, intr_name="llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x32_bf16(a: v4i16, b: v8i16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3179, intr_name="llvm.amdgcn.smfmac.f32.16x16x32.bf16", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x32_f16(a: v4f16, b: v8f16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3180, intr_name="llvm.amdgcn.smfmac.f32.16x16x32.f16", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_bf16(a: v8bf16, b: v16bf16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3181, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.bf16", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_bf8_bf8(a: v2i32, b: v4i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3182, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.bf8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_bf8_fp8(a: v2i32, b: v4i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3183, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.bf8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_f16(a: v8f16, b: v16f16, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3184, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.f16", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_fp8_bf8(a: v2i32, b: v4i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3185, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.fp8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_16x16x64_fp8_fp8(a: v2i32, b: v4i32, c: v4f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3186, intr_name="llvm.amdgcn.smfmac.f32.16x16x64.fp8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x16_bf16(a: v4i16, b: v8i16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3187, intr_name="llvm.amdgcn.smfmac.f32.32x32x16.bf16", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x16_f16(a: v4f16, b: v8f16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3188, intr_name="llvm.amdgcn.smfmac.f32.32x32x16.f16", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_bf16(a: v8bf16, b: v16bf16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3189, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.bf16", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_bf8_bf8(a: v2i32, b: v4i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3190, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.bf8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_bf8_fp8(a: v2i32, b: v4i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3191, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.bf8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_f16(a: v8f16, b: v16f16, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3192, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.f16", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_fp8_bf8(a: v2i32, b: v4i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3193, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.fp8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x32_fp8_fp8(a: v2i32, b: v4i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3194, intr_name="llvm.amdgcn.smfmac.f32.32x32x32.fp8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x64_bf8_bf8(a: v4i32, b: v8i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3195, intr_name="llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x64_bf8_fp8(a: v4i32, b: v8i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3196, intr_name="llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x64_fp8_bf8(a: v4i32, b: v8i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3197, intr_name="llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8", is_overloaded=False, return_type=return_type)


def smfmac_f32_32x32x64_fp8_fp8(a: v4i32, b: v8i32, c: v16f32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3198, intr_name="llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8", is_overloaded=False, return_type=return_type)


def smfmac_i32_16x16x128_i8(a: v4i32, b: v8i32, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3199, intr_name="llvm.amdgcn.smfmac.i32.16x16x128.i8", is_overloaded=False, return_type=return_type)


def smfmac_i32_16x16x64_i8(a: v2i32, b: v4i32, c: v4i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3200, intr_name="llvm.amdgcn.smfmac.i32.16x16x64.i8", is_overloaded=False, return_type=return_type)


def smfmac_i32_32x32x32_i8(a: v2i32, b: v4i32, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3201, intr_name="llvm.amdgcn.smfmac.i32.32x32x32.i8", is_overloaded=False, return_type=return_type)


def smfmac_i32_32x32x64_i8(a: v4i32, b: v8i32, c: v16i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3202, intr_name="llvm.amdgcn.smfmac.i32.32x32x64.i8", is_overloaded=False, return_type=return_type)


def softwqm(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3203, intr_name="llvm.amdgcn.softwqm", is_overloaded=True, return_type=return_type)


def sqrt(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3204, intr_name="llvm.amdgcn.sqrt", is_overloaded=True, return_type=return_type)


def strict_wqm(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3205, intr_name="llvm.amdgcn.strict.wqm", is_overloaded=True, return_type=return_type)


def strict_wwm(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3206, intr_name="llvm.amdgcn.strict.wwm", is_overloaded=True, return_type=return_type)


def struct_atomic_buffer_load(a: v4i32, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3207, intr_name="llvm.amdgcn.struct.atomic.buffer.load", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_add(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3208, intr_name="llvm.amdgcn.struct.buffer.atomic.add", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_and(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3209, intr_name="llvm.amdgcn.struct.buffer.atomic.and", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_cmpswap(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: v4i32, d: i32, e: i32, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=3210, intr_name="llvm.amdgcn.struct.buffer.atomic.cmpswap", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_cond_sub_u32(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3211, intr_name="llvm.amdgcn.struct.buffer.atomic.cond.sub.u32", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_dec(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3212, intr_name="llvm.amdgcn.struct.buffer.atomic.dec", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_fadd(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3213, intr_name="llvm.amdgcn.struct.buffer.atomic.fadd", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_fmax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3214, intr_name="llvm.amdgcn.struct.buffer.atomic.fmax", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_fmin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3215, intr_name="llvm.amdgcn.struct.buffer.atomic.fmin", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_inc(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3216, intr_name="llvm.amdgcn.struct.buffer.atomic.inc", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_or(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3217, intr_name="llvm.amdgcn.struct.buffer.atomic.or", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_smax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3218, intr_name="llvm.amdgcn.struct.buffer.atomic.smax", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_smin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3219, intr_name="llvm.amdgcn.struct.buffer.atomic.smin", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_sub(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3220, intr_name="llvm.amdgcn.struct.buffer.atomic.sub", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_swap(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3221, intr_name="llvm.amdgcn.struct.buffer.atomic.swap", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_umax(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3222, intr_name="llvm.amdgcn.struct.buffer.atomic.umax", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_umin(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3223, intr_name="llvm.amdgcn.struct.buffer.atomic.umin", is_overloaded=True, return_type=return_type)


def struct_buffer_atomic_xor(a: LLVMMatchType[Literal[0]], b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3224, intr_name="llvm.amdgcn.struct.buffer.atomic.xor", is_overloaded=True, return_type=return_type)


def struct_buffer_load(a: v4i32, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3225, intr_name="llvm.amdgcn.struct.buffer.load", is_overloaded=True, return_type=return_type)


def struct_buffer_load_format(a: v4i32, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3226, intr_name="llvm.amdgcn.struct.buffer.load.format", is_overloaded=True, return_type=return_type)


def struct_buffer_load_lds(a: v4i32, b: LLVMQualPointerType[Literal[3]], c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=3227, intr_name="llvm.amdgcn.struct.buffer.load.lds", is_overloaded=False, return_type=return_type)


def struct_buffer_store(a: any, b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3228, intr_name="llvm.amdgcn.struct.buffer.store", is_overloaded=True, return_type=return_type)


def struct_buffer_store_format(a: any, b: v4i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3229, intr_name="llvm.amdgcn.struct.buffer.store.format", is_overloaded=True, return_type=return_type)


def struct_ptr_atomic_buffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3230, intr_name="llvm.amdgcn.struct.ptr.atomic.buffer.load", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_add(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3231, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.add", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_and(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3232, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.and", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_cmpswap(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: AMDGPUBufferRsrcTy, d: i32, e: i32, f: i32, g: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=3233, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_cond_sub_u32(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3234, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.cond.sub.u32", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_dec(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3235, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.dec", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_fadd(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3236, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.fadd", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_fmax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3237, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.fmax", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_fmin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3238, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.fmin", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_inc(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3239, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.inc", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_or(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3240, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.or", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_smax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3241, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.smax", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_smin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3242, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.smin", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_sub(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3243, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.sub", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_swap(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3244, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.swap", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_umax(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3245, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.umax", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_umin(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3246, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.umin", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_atomic_xor(a: LLVMMatchType[Literal[0]], b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3247, intr_name="llvm.amdgcn.struct.ptr.buffer.atomic.xor", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3248, intr_name="llvm.amdgcn.struct.ptr.buffer.load", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_load_format(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, e: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, intr_id=3249, intr_name="llvm.amdgcn.struct.ptr.buffer.load.format", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_load_lds(a: AMDGPUBufferRsrcTy, b: LLVMQualPointerType[Literal[3]], c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, h, intr_id=3250, intr_name="llvm.amdgcn.struct.ptr.buffer.load.lds", is_overloaded=False, return_type=return_type)


def struct_ptr_buffer_store(a: any, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3251, intr_name="llvm.amdgcn.struct.ptr.buffer.store", is_overloaded=True, return_type=return_type)


def struct_ptr_buffer_store_format(a: any, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, intr_id=3252, intr_name="llvm.amdgcn.struct.ptr.buffer.store.format", is_overloaded=True, return_type=return_type)


def struct_ptr_tbuffer_load(a: AMDGPUBufferRsrcTy, b: i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3253, intr_name="llvm.amdgcn.struct.ptr.tbuffer.load", is_overloaded=True, return_type=return_type)


def struct_ptr_tbuffer_store(a: any, b: AMDGPUBufferRsrcTy, c: i32, d: i32, e: i32, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=3254, intr_name="llvm.amdgcn.struct.ptr.tbuffer.store", is_overloaded=True, return_type=return_type)


def struct_tbuffer_load(a: v4i32, b: i32, c: i32, d: i32, e: i32, f: i32, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3255, intr_name="llvm.amdgcn.struct.tbuffer.load", is_overloaded=True, return_type=return_type)


def struct_tbuffer_store(a: any, b: v4i32, c: i32, d: i32, e: i32, f: i32, g: i32, return_type=None):
    call_intrinsic(a, b, c, d, e, f, g, intr_id=3256, intr_name="llvm.amdgcn.struct.tbuffer.store", is_overloaded=True, return_type=return_type)


def sudot4(a: i1, b: i32, c: i1, d: i32, e: i32, f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3257, intr_name="llvm.amdgcn.sudot4", is_overloaded=False, return_type=return_type)


def sudot8(a: i1, b: i32, c: i1, d: i32, e: i32, f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3258, intr_name="llvm.amdgcn.sudot8", is_overloaded=False, return_type=return_type)


def swmmac_bf16_16x16x32_bf16(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3259, intr_name="llvm.amdgcn.swmmac.bf16.16x16x32.bf16", is_overloaded=True, return_type=return_type)


def swmmac_f16_16x16x32_f16(a: anyfloat, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3260, intr_name="llvm.amdgcn.swmmac.f16.16x16x32.f16", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_bf16(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3261, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.bf16", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_bf8_bf8(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3262, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_bf8_fp8(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3263, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_f16(a: anyfloat, b: anyfloat, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3264, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.f16", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_fp8_bf8(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3265, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8", is_overloaded=True, return_type=return_type)


def swmmac_f32_16x16x32_fp8_fp8(a: anyint, b: anyint, c: LLVMMatchType[Literal[0]], d: anyint, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3266, intr_name="llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8", is_overloaded=True, return_type=return_type)


def swmmac_i32_16x16x32_iu4(a: i1, b: anyint, c: i1, d: anyint, e: LLVMMatchType[Literal[0]], f: anyint, g: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=3267, intr_name="llvm.amdgcn.swmmac.i32.16x16x32.iu4", is_overloaded=True, return_type=return_type)


def swmmac_i32_16x16x32_iu8(a: i1, b: anyint, c: i1, d: anyint, e: LLVMMatchType[Literal[0]], f: anyint, g: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=3268, intr_name="llvm.amdgcn.swmmac.i32.16x16x32.iu8", is_overloaded=True, return_type=return_type)


def swmmac_i32_16x16x64_iu4(a: i1, b: anyint, c: i1, d: anyint, e: LLVMMatchType[Literal[0]], f: anyint, g: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, g, intr_id=3269, intr_name="llvm.amdgcn.swmmac.i32.16x16x64.iu4", is_overloaded=True, return_type=return_type)


def trig_preop(a: LLVMMatchType[Literal[0]], b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3270, intr_name="llvm.amdgcn.trig.preop", is_overloaded=True, return_type=return_type)


def ubfe(a: LLVMMatchType[Literal[0]], b: i32, c: i32, return_type=None):
    return call_intrinsic(a, b, c, intr_id=3271, intr_name="llvm.amdgcn.ubfe", is_overloaded=True, return_type=return_type)


def udot2(a: v2i16, b: v2i16, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3272, intr_name="llvm.amdgcn.udot2", is_overloaded=False, return_type=return_type)


def udot4(a: i32, b: i32, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3273, intr_name="llvm.amdgcn.udot4", is_overloaded=False, return_type=return_type)


def udot8(a: i32, b: i32, c: i32, d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3274, intr_name="llvm.amdgcn.udot8", is_overloaded=False, return_type=return_type)


def unreachable(return_type=None):
    call_intrinsic(intr_id=3275, intr_name="llvm.amdgcn.unreachable", is_overloaded=False, return_type=return_type)


def update_dpp(a: LLVMMatchType[Literal[0]], b: LLVMMatchType[Literal[0]], c: i32, d: i32, e: i32, f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3276, intr_name="llvm.amdgcn.update.dpp", is_overloaded=True, return_type=return_type)


def wave_barrier(return_type=None):
    call_intrinsic(intr_id=3277, intr_name="llvm.amdgcn.wave.barrier", is_overloaded=False, return_type=return_type)


def wave_id(return_type=None):
    return call_intrinsic(intr_id=3278, intr_name="llvm.amdgcn.wave.id", is_overloaded=False, return_type=return_type)


def wave_reduce_umax(a: LLVMMatchType[Literal[0]], b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3279, intr_name="llvm.amdgcn.wave.reduce.umax", is_overloaded=True, return_type=return_type)


def wave_reduce_umin(a: LLVMMatchType[Literal[0]], b: i32, return_type=None):
    return call_intrinsic(a, b, intr_id=3280, intr_name="llvm.amdgcn.wave.reduce.umin", is_overloaded=True, return_type=return_type)


def wavefrontsize(return_type=None):
    return call_intrinsic(intr_id=3281, intr_name="llvm.amdgcn.wavefrontsize", is_overloaded=False, return_type=return_type)


def wmma_bf16_16x16x16_bf16(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]], d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3282, intr_name="llvm.amdgcn.wmma.bf16.16x16x16.bf16", is_overloaded=True, return_type=return_type)


def wmma_bf16_16x16x16_bf16_tied(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]], d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3283, intr_name="llvm.amdgcn.wmma.bf16.16x16x16.bf16.tied", is_overloaded=True, return_type=return_type)


def wmma_f16_16x16x16_f16(a: anyfloat, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]], d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3284, intr_name="llvm.amdgcn.wmma.f16.16x16x16.f16", is_overloaded=True, return_type=return_type)


def wmma_f16_16x16x16_f16_tied(a: anyfloat, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]], d: i1, return_type=None):
    return call_intrinsic(a, b, c, d, intr_id=3285, intr_name="llvm.amdgcn.wmma.f16.16x16x16.f16.tied", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_bf16(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3286, intr_name="llvm.amdgcn.wmma.f32.16x16x16.bf16", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_bf8_bf8(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3287, intr_name="llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_bf8_fp8(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3288, intr_name="llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_f16(a: anyfloat, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3289, intr_name="llvm.amdgcn.wmma.f32.16x16x16.f16", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_fp8_bf8(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3290, intr_name="llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8", is_overloaded=True, return_type=return_type)


def wmma_f32_16x16x16_fp8_fp8(a: anyint, b: LLVMMatchType[Literal[1]], c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3291, intr_name="llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8", is_overloaded=True, return_type=return_type)


def wmma_i32_16x16x16_iu4(a: i1, b: anyint, c: i1, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[0]], f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3292, intr_name="llvm.amdgcn.wmma.i32.16x16x16.iu4", is_overloaded=True, return_type=return_type)


def wmma_i32_16x16x16_iu8(a: i1, b: anyint, c: i1, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[0]], f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3293, intr_name="llvm.amdgcn.wmma.i32.16x16x16.iu8", is_overloaded=True, return_type=return_type)


def wmma_i32_16x16x32_iu4(a: i1, b: anyint, c: i1, d: LLVMMatchType[Literal[1]], e: LLVMMatchType[Literal[0]], f: i1, return_type=None):
    return call_intrinsic(a, b, c, d, e, f, intr_id=3294, intr_name="llvm.amdgcn.wmma.i32.16x16x32.iu4", is_overloaded=True, return_type=return_type)


def workgroup_id_x(return_type=None):
    return call_intrinsic(intr_id=3295, intr_name="llvm.amdgcn.workgroup.id.x", is_overloaded=False, return_type=return_type)


def workgroup_id_y(return_type=None):
    return call_intrinsic(intr_id=3296, intr_name="llvm.amdgcn.workgroup.id.y", is_overloaded=False, return_type=return_type)


def workgroup_id_z(return_type=None):
    return call_intrinsic(intr_id=3297, intr_name="llvm.amdgcn.workgroup.id.z", is_overloaded=False, return_type=return_type)


def workitem_id_x(return_type=None):
    return call_intrinsic(intr_id=3298, intr_name="llvm.amdgcn.workitem.id.x", is_overloaded=False, return_type=return_type)


def workitem_id_y(return_type=None):
    return call_intrinsic(intr_id=3299, intr_name="llvm.amdgcn.workitem.id.y", is_overloaded=False, return_type=return_type)


def workitem_id_z(return_type=None):
    return call_intrinsic(intr_id=3300, intr_name="llvm.amdgcn.workitem.id.z", is_overloaded=False, return_type=return_type)


def wqm(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3301, intr_name="llvm.amdgcn.wqm", is_overloaded=True, return_type=return_type)


def wqm_demote(a: i1, return_type=None):
    call_intrinsic(a, intr_id=3302, intr_name="llvm.amdgcn.wqm.demote", is_overloaded=False, return_type=return_type)


def wqm_vote(a: i1, return_type=None):
    return call_intrinsic(a, intr_id=3303, intr_name="llvm.amdgcn.wqm.vote", is_overloaded=False, return_type=return_type)


def writelane(a: LLVMMatchType[Literal[0]], b: i32, c: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, b, c, intr_id=3304, intr_name="llvm.amdgcn.writelane", is_overloaded=True, return_type=return_type)


def wwm(a: LLVMMatchType[Literal[0]]):
    return call_intrinsic(a, intr_id=3305, intr_name="llvm.amdgcn.wwm", is_overloaded=True, return_type=return_type)
