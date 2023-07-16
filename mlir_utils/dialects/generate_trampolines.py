import ast
import copy
import inspect
import keyword
from pathlib import Path
from textwrap import dedent

import black
import inflection


def ast_call(name, args=None, keywords=None):
    if keywords is None:
        keywords = []
    if args is None:
        args = []
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=args,
        keywords=keywords,
    )


def make_fun(op_name, op_class):
    _mod = ast.parse(dedent(inspect.getsource(op_class.__init__)))
    init_fn = next(n for n in _mod.body if isinstance(n, ast.FunctionDef))
    args = init_fn.args
    args.args.pop(0)
    for a in args.args:
        a.arg = inflection.underscore(a.arg).lower()

    for k in args.kwonlyargs:
        k.arg = inflection.underscore(k.arg).lower()

    keywords = [
        ast.keyword(k.arg, ast.Name(k.arg))
        for k, d in zip(args.kwonlyargs, args.kw_defaults)
    ]

    for a in args.args + args.kwonlyargs:
        a.annotation = None
    ret = ast.parse(
        f"return get_result_or_results({ast.unparse(ast_call(op_name, args.args, keywords))})"
    ).body[0]
    fun_name = op_class.OPERATION_NAME.split(".")[-1]
    if keyword.iskeyword(fun_name):
        fun_name = fun_name + "_"
    n = ast.FunctionDef(
        name=fun_name,
        args=copy.deepcopy(args),
        body=[ret],
        decorator_list=[],
    )
    ast.fix_missing_locations(n)
    return n


def generate_trampoline(input_module, output_file_path, skips=None):
    import mlir_utils
    from mlir_utils.dialects.util import get_result_or_results

    if skips is None:
        skips = set()
    skips.update({"_Dialect"})
    init_funs = {}
    for name, obj in inspect.getmembers(input_module):
        if (
            inspect.isclass(obj)
            and hasattr(obj, "OPERATION_NAME")
            and obj.__name__ not in skips
        ):
            if not inspect.isfunction(obj.__init__):
                # some builders don't have any __init__ but inherit from opview
                continue
            init_funs[obj.__name__] = obj

    if not len(init_funs):
        return

    functions = [make_fun(n, s) for n, s in init_funs.items()]

    ods_imports = ast.ImportFrom(
        module=mlir_utils.dialects.util.__name__,
        names=[ast.alias(get_result_or_results.__name__)],
        level=0,
    )
    op_imports = ast.ImportFrom(
        module=input_module.__name__,
        names=[ast.alias(n) for n in init_funs.keys()],
        level=0,
    )
    print(f"generating for {input_module.__name__}")
    if "dialects.linalg" in input_module.__name__:
        linalg_imports = [
            ast.parse(
                f"from {input_module.__name__} import TensorExpression, SymbolDef, DimDef, AffineExprDef"
            ).body[0],
            ast.parse("from typing import Tuple").body[0],
        ]
    else:
        linalg_imports = []

    new_mod = ast.Module([op_imports, *linalg_imports, ods_imports] + functions, [])
    new_src = ast.unparse(new_mod)
    formatted_new_src = black.format_file_contents(
        new_src, fast=False, mode=black.Mode()
    )
    output_file = open(output_file_path, "w")
    output_file.write(formatted_new_src)


def generate_all_upstream_trampolines():
    import mlir.dialects

    # noinspection PyUnresolvedReferences
    from mlir.dialects import (
        arith,
        async_dialect,
        bufferization,
        builtin,
        cf,
        complex,
        func,
        gpu,
        linalg,
        math,
        memref,
        ml_program,
        pdl,
        quant,
        scf,
        shape,
        sparse_tensor,
        tensor,
        tosa,
        transform,
        vector,
    )

    for name, mod in inspect.getmembers(mlir.dialects, inspect.ismodule):
        if not name.startswith("_"):
            generate_trampoline(mod, Path(__file__).parent / (name + ".py"))
