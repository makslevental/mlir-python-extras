from mlir.extras.runtime.passes import Pipeline as pipe


def test_basic():
    p = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
        .cse()
        .lower_affine()
        .Func(pipe().convert_arith_to_llvm())
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

    p1 = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
    )

    p2 = (
        pipe()
        .cse()
        .lower_affine()
        .Func(pipe().convert_arith_to_llvm())
        .convert_func_to_llvm()
        .canonicalize()
        .convert_openmp_to_llvm()
        .cse()
        .reconcile_unrealized_casts()
    )

    assert (
        p1 + p2
    ).materialize() == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"

    p1 = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
    )
    assert (
        str(p1)
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm)"
    )

    p1 += p2
    assert (
        str(p1)
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"
    )


def test_context():
    p = pipe().Context(
        "aie.device",
        pipe().add_pass("aie-localize-locks").add_pass("aie-normalize-address-spaces"),
    )
    assert (
        str(p)
        == "builtin.module(aie.device(aie-localize-locks,aie-normalize-address-spaces))"
    )
