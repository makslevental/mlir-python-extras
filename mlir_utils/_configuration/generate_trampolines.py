import argparse
import ast
import copy
import inspect
import keyword
import os
import pkgutil
from pathlib import Path
from textwrap import dedent

import black
import inflection


def ast_call(name, args=None, keywords=None):
    if keywords is None:
        keywords = []
    if args is None:
        args = []
    call = ast.Call(
        func=ast.Name(name),
        args=args,
        keywords=keywords,
    )
    return call


class FindOperands(ast.NodeVisitor):
    def __init__(self):
        self.operands = {}
        self.results = {}

    def visit_Call(self, node: ast.Call):
        if hasattr(node.func, "value") and hasattr(node.func.value, "id"):
            if node.func.value.id == "operands":
                if isinstance(node.args[0], ast.Call):
                    nested_call = node.args[0]
                    is_optional = False
                elif isinstance(node.args[0], ast.IfExp):
                    nested_call = node.args[0].body
                    is_optional = True
                else:
                    raise RuntimeError(
                        f"unsupported operands python code: {ast.unparse(node)}"
                    )
                oper_name = inflection.underscore(nested_call.args[0].id).lower()
                is_variadic = "values" in nested_call.func.id
                type = "list[Value]" if is_variadic else "Value"
                if is_optional:
                    type = f"Optional[{type}]"
                self.operands[oper_name] = type
            elif node.func.value.id == "results":
                if node.func.attr == "extend":
                    if isinstance(node.args[0], ast.BinOp):
                        # something like results.extend([operands[0].type] * 1)
                        return
                    else:
                        self.results[node.args[0].id] = "list[Type]"
                elif node.func.attr == "append":
                    self.results[node.args[0].id] = "Type"
                else:
                    raise ValueError("unknown results object")


# TODO(max): ops that have symboltables need to be classes but that requires some upstream support for statically
# identifying such ops
def generate_op_trampoline(op_class):
    from mlir_utils.util import (
        get_result_or_results,
        maybe_cast,
        region_op,
        get_user_code_loc,
    )

    _mod = ast.parse(dedent(inspect.getsource(op_class.__init__)))
    init_fn = next(n for n in _mod.body if isinstance(n, ast.FunctionDef))
    args = init_fn.args
    args.args.pop(0)
    for a in args.args:
        a.arg = inflection.underscore(a.arg).lower()

    fun_name = (
        op_class.OPERATION_NAME.split(".", 1)[-1].replace(".", "_").replace("-", "_")
    )
    if keyword.iskeyword(fun_name):
        fun_name = fun_name + "_"
    op_class_name = op_class.__name__
    body = [ast.parse(f"if loc is None: loc = {get_user_code_loc.__name__}()")]
    if len(args.args) == 1 and args.args[0].arg == "results_":
        args.defaults.append(ast.Constant(None))
        body += [ast.parse("results_ = results_ or []").body[0]]

    keywords = [
        ast.keyword(k.arg, ast.Name(inflection.underscore(k.arg).lower()))
        for k, d in zip(args.kwonlyargs, args.kw_defaults)
    ]
    if (
        hasattr(op_class, "_ODS_REGIONS")
        and op_class._ODS_REGIONS[0] == 1
        and not op_class.OPERATION_NAME.startswith("linalg")
    ):
        decorator_list = [ast.Name(id=region_op.__name__, ctx=ast.Load())]
        body += [ast.Return([ast_call(op_class_name, args.args, keywords)])]
    else:
        decorator_list = []
        body += [
            ast.parse(
                f"return {maybe_cast.__name__}({get_result_or_results.__name__}({ast.unparse(ast_call(op_class_name, args.args, keywords))}))"
            ).body[0]
        ]

    for k in args.kwonlyargs:
        k.arg = inflection.underscore(k.arg).lower()

    args = copy.deepcopy(args)
    oper_finder = FindOperands()
    oper_finder.visit(init_fn)
    for a in args.args:
        if a.arg in oper_finder.operands:
            a.annotation = ast.Name(id=oper_finder.operands[a.arg], ctx=ast.Load())
        elif a.arg in oper_finder.results:
            a.annotation = ast.Name(id=oper_finder.results[a.arg], ctx=ast.Load())
    n = ast.FunctionDef(
        name=fun_name,
        args=args,
        body=body,
        decorator_list=decorator_list,
    )
    ast.fix_missing_locations(n)
    return n


def generate_dialect_trampolines_from_module(input_module, skips: set):
    import mlir_utils
    from mlir_utils.util import (
        get_result_or_results,
        maybe_cast,
        region_op,
        get_user_code_loc,
    )
    import mlir.dialects._ods_common
    from mlir_utils._configuration.configuration import _get_mlir_package_prefix

    skips.update({"_Dialect"})
    init_funs = {}
    for name, obj in inspect.getmembers(input_module):
        if (
            inspect.isclass(obj)
            and hasattr(obj, "OPERATION_NAME")
            and obj.__name__ not in skips
        ):
            if obj.__module__ == mlir.dialects._ods_common.__name__:
                # these are extension classes and we should wrap the generated class instead
                obj = obj.__base__
            if not inspect.isfunction(obj.__init__):
                print(f"skipping {obj.__name__} because it has no __init__")
                # some builders don't have any __init__ but inherit from opview
                continue
            init_funs[obj.__name__] = obj

    if not len(init_funs):
        return

    functions = [
        generate_op_trampoline(op_class)
        for op_class in sorted(init_funs.values(), key=lambda o: o.__name__)
    ]

    ir_imports = ast.ImportFrom(
        module=_get_mlir_package_prefix() + ".ir",
        names=[ast.alias(i) for i in ["Value", "Attribute", "Type"]],
        level=0,
    )
    ods_imports = ast.ImportFrom(
        module=mlir_utils.util.__name__,
        names=[
            ast.alias(f.__name__)
            for f in [get_result_or_results, maybe_cast, region_op, get_user_code_loc]
        ],
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

    all = ast.parse(f"__all__ = [{', '.join(repr(f.name) for f in functions)}]")

    new_mod = ast.Module(
        [ir_imports, op_imports, *linalg_imports, ods_imports] + functions + [all], []
    )
    new_src = ast.unparse(new_mod)
    return black.format_file_contents(new_src, fast=False, mode=black.Mode())


def generate_trampolines(
    mod_path=None, dst_path: Path = None, output_name=None, skips: set = None
):
    from mlir_utils._configuration.configuration import (
        _add_file_to_sources_txt_file,
        PACKAGE_ROOT_PATH,
    )
    import mlir_utils.dialects

    if mod_path is None:
        parser = argparse.ArgumentParser(
            prog="generate-trampolines",
            description="Generate trampolines for a particular module",
        )
        parser.add_argument("mod_path", type=str)
        parser.add_argument("-d", "--dst-path", type=Path, required=False)
        parser.add_argument("-n", "--name", required=False)
        parser.add_argument("--skips", required=False, nargs="*", type=set)
        args = parser.parse_args()
        mod_path = args.mod_path
        if args.dst_path:
            dst_path = args.dst_path
        if args.skips:
            skips = set(args.skips)
        if args.name:
            output_name = args.name

    if dst_path is None:
        dst_path = Path(mlir_utils.dialects.__path__[0])

    assert dst_path.is_dir(), "destination path isn't a directory or doesn't exist"
    if skips is None:
        skips = set()
    if output_name is None:
        output_name = mod_path.rsplit(".", maxsplit=1)[-1]

    try:
        # you need the star here to import the whole submodule path rather than just the root module (mlir)
        modu = __import__(mod_path, fromlist=["*"])
    except (ImportError, ModuleNotFoundError) as e:
        print(f"couldn't import or find module {mod_path}")
        raise e

    if generated := generate_dialect_trampolines_from_module(modu, skips):
        dst_path = dst_path / f"{output_name}.py"
        with open(dst_path, "w") as f:
            f.write(generated)
        if dst_path.is_relative_to(PACKAGE_ROOT_PATH):
            _add_file_to_sources_txt_file(dst_path)


def generate_linalg(mod_path):
    import mlir_utils.dialects
    import mlir.dialects
    from mlir_utils._configuration.configuration import _add_file_to_sources_txt_file
    from mlir.dialects.linalg import DefinedOpCallable, OperandKind
    from mlir_utils.util import (
        get_result_or_results,
        maybe_cast,
        region_op,
        get_user_code_loc,
    )

    linalg_modu = __import__(mod_path, fromlist=["*"])

    dst_path = Path(mlir_utils.dialects.__path__[0])
    dst_path = dst_path / f"linalg.py"
    with open(dst_path, "w") as f:
        functions = []
        linalg_import = ast.ImportFrom(
            module=mlir.dialects.__name__,
            names=[ast.Name("linalg")],
            level=0,
        )
        ods_imports = ast.ImportFrom(
            module=mlir_utils.util.__name__,
            names=[ast.alias(f.__name__) for f in [maybe_cast, get_user_code_loc]],
            level=0,
        )
        _keywords = [
            ast.keyword("loc", ast.Name("loc")),
            ast.keyword("ip", ast.Name("ip")),
        ]
        for name, op_callable in inspect.getmembers(
            linalg_modu, lambda x: isinstance(x, DefinedOpCallable)
        ):
            inputs = [
                ast.Name(o)
                for o, def_ in op_callable.op_def.registered_operands.items()
                if def_.kind == OperandKind.INPUT_TENSOR
            ]
            outputs = [
                ast.Name(o)
                for o, def_ in op_callable.op_def.registered_operands.items()
                if def_.kind == OperandKind.OUTPUT_TENSOR
            ]

            keywords = _keywords + [ast.keyword("outs", ast.List(outputs))]
            # body = [ast.Str(op_callable.op_def.metadata.doc)]
            body = [ast.parse(f"if loc is None: loc = {get_user_code_loc.__name__}()")]
            body += [
                ast.parse(
                    f"return {maybe_cast.__name__}({ast.unparse(ast_call('linalg.' + name, inputs, keywords))})"
                ).body[0]
            ]
            n = ast.FunctionDef(
                name=op_callable.op_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=inputs + outputs,
                    defaults=[],
                    kwonlyargs=[ast.Name("loc"), ast.Name("ip")],
                    kw_defaults=[ast.Name("None"), ast.Name("None")],
                ),
                body=body,
                decorator_list=[],
            )
            ast.fix_missing_locations(n)
            functions.append(n)

        new_mod = ast.Module([linalg_import, ods_imports] + functions, [])
        new_src = ast.unparse(new_mod)
        generated = black.format_file_contents(new_src, fast=False, mode=black.Mode())
        f.write(generated)
    _add_file_to_sources_txt_file(dst_path)


def generate_all_upstream_trampolines():
    import mlir.dialects
    import mlir_utils.dialects

    mlir_mod_path = Path(mlir_utils.dialects.__path__[0])
    if not mlir_mod_path.exists():
        os.makedirs(mlir_mod_path)

    for mod in pkgutil.iter_modules(mlir.dialects.__path__):
        if not mod.name.startswith("_") and mod.name not in {"linalg", "transform"}:
            generate_trampolines(
                f"{mlir.dialects.__name__}.{mod.name}",
                mlir_mod_path,
                mod.name,
                skips={"1kOp"},
            )
        elif mod.name == "linalg":
            generate_linalg(f"{mlir.dialects.__name__}.linalg")
        elif mod.name == "transform":
            transform_module_path = Path(mlir.dialects.__path__[0]) / mod.name
            transform_module_path_mlir_utils = mlir_mod_path / mod.name
            if not transform_module_path_mlir_utils.exists():
                os.makedirs(transform_module_path_mlir_utils)

            generate_trampolines(
                f"{mlir.dialects.__name__}.{mod.name}",
                transform_module_path_mlir_utils,
                "__init__",
                skips={"1kOp"},
            )
            for mod in pkgutil.iter_modules([str(transform_module_path)]):
                generate_trampolines(
                    f"{mlir.dialects.__name__}.transform.{mod.name}",
                    transform_module_path_mlir_utils,
                    mod.name,
                    skips={"1kOp"},
                )
