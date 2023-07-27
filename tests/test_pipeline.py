from mlir_utils.runtime.passes import cse, lower_affine, convert_arith_to_llvm


def test_basic():
    p = (
        cse()
        .Func(lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
        .cse()
        .lower_affine()
        .Func(convert_arith_to_llvm())
        .convert_func_to_llvm()
        .canonicalize()
        .convert_openmp_to_llvm()
        .cse()
        .reconcile_unrealized_casts()
    )
    assert (
        p.materialize()
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"
    )
