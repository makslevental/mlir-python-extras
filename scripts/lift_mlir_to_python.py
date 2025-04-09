import argparse
import inspect
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
from mlir.dialects import builtin, func
from mlir.ir import (
    WalkResult,
    WalkOrder,
    IntegerAttr,
    DenseI64ArrayAttr,
    DenseI32ArrayAttr,
    BoolAttr,
    FlatSymbolRefAttr,
    Attribute,
    ArrayAttr,
    DenseIntElementsAttr,
    DenseFPElementsAttr,
    FloatAttr,
    StringAttr,
    TypeAttr,
    IndexType,
    ShapedType,
    Value,
    OpView,
    OpOperandList,
    IntegerType,
    F32Type,
    F16Type,
    F64Type,
)

from mlir.extras.context import mlir_mod_ctx
from mlir.extras.dialects.ext import arith
from mlir.extras.dialects.ext import scf

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir.extras.util import mlir_type_to_np_dtype


INDENT = 0
# OUTPUT_BUF = io.StringIO()
OUTPUT_BUF = sys.stdout
ATTR_ALIASES = {}


def normalize_ssa(ssa: str | Value):
    if isinstance(ssa, Value):
        ssa = ssa.get_name(use_name_loc_as_prefix=True)
    if ssa[1].isnumeric():
        ssa = ssa.replace("%", "v")
    else:
        ssa = ssa.replace("%", "")
    ssa = ssa.replace("-", "_")
    ssa = ssa.replace("#", "_")
    return ssa


def normalize_op_name(name: str):
    name = name.replace("(", "_").replace(")", "_").replace(", ", "_").replace(",", "_")
    split_on_dots = name.split(".")
    if len(split_on_dots) > 2:
        dialect, op = split_on_dots[0], "_".join(split_on_dots[1:])
        split_on_dots = [dialect, op]
    return ".".join(split_on_dots)


def np_array_from_shape_type(shape, dtype, splat_value=None):
    if splat_value:
        return np.full(shape, splat_value, dtype)
    return np.empty(shape, dtype)


_dense_i_array_attr_reg = re.compile(r"array<i\d+: (.*?)>")
_integer_overflow_flags_reg = re.compile(r"#arith.overflow<(.*?)>")


def map_attr(attr):
    if attr in ATTR_ALIASES:
        return ATTR_ALIASES[attr]
    attr = attr.maybe_downcast()
    if isinstance(attr, (IntegerAttr, BoolAttr, FloatAttr)):
        return attr.value
    if isinstance(attr, (FlatSymbolRefAttr, StringAttr)):
        return repr(attr.value)
    # TODO(max): add things upstream to query ArrayAttr elements via .value
    if isinstance(attr, (DenseI32ArrayAttr, DenseI64ArrayAttr)):
        s = str(attr)
        elements = _dense_i_array_attr_reg.findall(s)
        if len(elements) == 0:
            return "[]"
        return f"[{elements[0]}]"
    if isinstance(attr, (DenseIntElementsAttr, DenseFPElementsAttr)):
        if attr.is_splat:
            splat_v = map_attr(attr.get_splat_value())
            # arr = np_array_from_shape_type(attr.type.shape, mlir_type_to_np_dtype(attr.type.element_type), splat_v)
            return f"np.full({attr.type.shape}, {splat_v}, np.{mlir_type_to_np_dtype(attr.type.element_type).__name__})"
    if attr.__class__ in {Attribute}:
        if "#arith.overflow" in str(attr):
            flag = _integer_overflow_flags_reg.findall(str(attr))
            assert len(flag) == 1
            return f"arith.IntegerOverflowFlags.{flag[0]}"
        return f"Attribute.parse('{attr}')"
    # TODO(max): add things upstream to query ArrayAttr elements via .value
    if attr.__class__ in {ArrayAttr}:
        return f"ArrayAttr.parse('{attr}')"
    if attr.__class__ in {TypeAttr}:
        return f"TypeAttr.parse('{attr}')"
    return f"Attribute.parse('{attr}')"


def map_type(type):
    type = type.maybe_downcast()
    if isinstance(type, (IntegerType, F16Type, F32Type, F64Type)):
        if type.width == 1:
            return f"T.bool()"
        return f"T.{type}()"
    if isinstance(type, ShapedType):
        encoding = ""
        if hasattr(type, "encoding") and type.encoding is not None:
            encoding = f", encoding={map_attr(type.encoding)}"
        type_name = str(type).split("<")[0]
        shape = [
            (
                "ShapedType.get_dynamic_size()"
                if s == ShapedType.get_dynamic_size()
                else str(s)
            )
            for s in type.shape
        ]
        return f"T.{type_name}({', '.join(shape)}, {map_type(type.element_type)}{encoding})"
    if isinstance(type, IndexType):
        return f"T.index()"
    return f"Type.parse('{type}')"


def get_init_args(opview):
    klass = opview.__class__
    while not klass.__base__ is OpView:
        klass = klass.__base__
    init_sig = inspect.getfullargspec(klass.__init__)
    init_args = init_sig.args[1:] + init_sig.kwonlyargs
    init_args.remove("loc")
    init_args.remove("ip")
    return init_args


def expects_result_first_arg(opview):
    klass = opview.__class__
    while not klass.__base__ is OpView:
        klass = klass.__base__
    init_sig = inspect.getfullargspec(klass.__init__)
    first_arg = init_sig.args[1]
    if first_arg in {"result"}:
        return first_arg


# stolen from inflection
def underscore(word: str) -> str:
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


opidx_counter = 0


def print_opview(opview, name=None):
    print("    " * INDENT, file=OUTPUT_BUF, end="")
    if len(opview.results):
        print(
            ", ".join([normalize_ssa(r) for r in opview.results]),
            end=" = ",
            file=OUTPUT_BUF,
        )

    if name is None:
        name = opview.name
    if isinstance(name, StringAttr):
        name = name.value
    name = normalize_op_name(name)

    attrs = {attr.name: attr.attr for attr in opview.attributes}
    op_idx_owner_name = None
    if "OpIdx" in attrs:
        global opidx_counter
        if len(opview.results):
            assert len(opview.results) == 1
            op_idx_owner_name = f"{normalize_ssa(opview.results[0])}"
        else:
            op_idx_owner_name = f"{name}_{opidx_counter}"
            print(op_idx_owner_name, end=" = ", file=OUTPUT_BUF)
        opidx_counter += 1

    print(f"{name}(", end="", file=OUTPUT_BUF)
    init_args = get_init_args(opview)
    operands_attrs = {}

    if init_args[0] in {"result"}:
        if len(opview.results) > 0:
            result = map_type(getattr(opview, init_args[0]).type)
            if isinstance(opview, func.CallOp):
                result = f"[{result}]"
        else:
            if isinstance(opview, func.CallOp):
                result = "[]"
            else:
                raise NotImplementedError

        operands_attrs["result"] = result
        init_args = init_args[1:]

    # using this causes a reference to the value to remain (causing LLVM ERROR: operation destroyed but still has uses at the end of the script)
    # results_ = {r for r in opview.results}
    results = {r.get_name() for r in opview.results}
    for oan in init_args:
        oa = getattr(opview, oan)
        py_oan = underscore(oan)
        if oa is None:
            continue
        if isinstance(oa, Value):
            if oa.get_name() not in results:
                operands_attrs[py_oan] = normalize_ssa(oa)
            else:
                assert len(
                    results
                ), "only single output result type currently supported"
                operands_attrs[py_oan] = map_type(oa.type)
        elif isinstance(oa, OpOperandList):
            operands_attrs[py_oan] = f"[{', '.join(normalize_ssa(o) for o in oa)}]"
        elif isinstance(oa, Attribute):
            operands_attrs[py_oan] = map_attr(oa.maybe_downcast())
        elif isinstance(oa, (int, float, bool)):
            operands_attrs[py_oan] = oa
        else:
            raise NotImplementedError(oa)
    print(
        ", ".join([f"{k}={v}" for k, v in operands_attrs.items()]),
        file=OUTPUT_BUF,
        end="",
    )
    print(f")", file=OUTPUT_BUF)

    if op_idx_owner_name is not None:
        if len(results):
            owner = f"{op_idx_owner_name}.owner"
        else:
            owner = f"{op_idx_owner_name}"
        print(
            "    " * INDENT
            + f"{owner}.attributes['OpIdx'] = amdgpu.OpIdxAttr.get({attrs['OpIdx'].value})",
            file=OUTPUT_BUF,
        )


def print_func_op(func_op: func.FuncOp):
    # op.print(print_generic_op_form=True)
    print("    " * INDENT, file=OUTPUT_BUF, end="")
    print("@func.func(", file=OUTPUT_BUF, end="")
    if len(func_op.attributes):
        attrs = []
        for i in range(len(func_op.attributes)):
            attr = func_op.attributes[i]
            if attr.name == "function_type":
                fun_type = attr.attr.value
                inputs = f"[{', '.join([map_type(t) for t in fun_type.inputs])}]"
                results = f"[{', '.join([map_type(t) for t in fun_type.results])}]"
                attrs.append(
                    f"{attr.name}=T.function(inputs={inputs}, results={results})"
                )
            else:
                attrs.append(f"{attr.name}={map_attr(attr.attr)}")
        print(", ".join(attrs), end="", file=OUTPUT_BUF)
    print(")", file=OUTPUT_BUF)
    args = list(func_op.body.blocks[0].arguments)
    args = list(map(normalize_ssa, args))
    print(
        f"def {normalize_op_name(func_op.sym_name.value)}({', '.join(args)}):",
        file=OUTPUT_BUF,
    )


def print_arith_constant(constop: arith.ConstantOp):
    print("    " * INDENT, file=OUTPUT_BUF, end="")
    print(
        f"{normalize_ssa(constop.result)} = arith.constant({map_attr(constop.value)}, {map_type(constop.result.type)})",
        file=OUTPUT_BUF,
    )


def print_scf_for(for_op: scf.ForOp):
    iv = normalize_ssa(for_op.induction_variable)
    iter_args = [normalize_ssa(a) for a in for_op.inner_iter_args]
    results = [normalize_ssa(r) for r in for_op.results]
    if len(iter_args) > 1:
        opers_str = f"{iv}, [{', '.join(iter_args)}], [{', '.join(results)}]"
    elif len(iter_args) == 1:
        opers_str = f"{iv}, {iter_args[0]}, {results[0]}"
    else:
        opers_str = f"{iv}"
    start, stop, step = map(
        normalize_ssa, [for_op.lowerBound, for_op.upperBound, for_op.step]
    )
    init_args = [normalize_ssa(a) for a in for_op.initArgs]
    print(
        ("    " * INDENT)
        + f"for {opers_str} in scf.for_({start}, {stop}, {step}, iter_args=[{', '.join(init_args)}]):",
        file=OUTPUT_BUF,
    )


def print_scf_if(if_op: scf.IfOp):
    assert len(if_op.results) == 1
    res = if_op.results[0]
    res_name = normalize_ssa(res)
    global INDENT

    def print_yield_as_return(yield_op: scf.YieldOp):
        opers = [normalize_ssa(a) for a in yield_op.operands]
        print(
            ("    " * INDENT) + f"return {', '.join(opers)}",
            file=OUTPUT_BUF,
        )

    print(
        textwrap.indent(
            textwrap.dedent(
                f"""\
                    @ext.scf.if_({normalize_ssa(if_op.condition)}, results=[{map_type(res.type)}])
                    def {res_name}():\
                """
            ),
            "    " * INDENT,
        ),
        file=OUTPUT_BUF,
    )
    INDENT += 1
    for bodyop in if_op.thenRegion.blocks[0].operations:
        if isinstance(bodyop, scf.YieldOp):
            print_yield_as_return(bodyop)
        else:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    INDENT -= 1
    print(
        textwrap.indent(
            textwrap.dedent(
                f"""\
                    @ext.scf.else_({res_name})
                    def {res_name}_else():\
                """,
            ),
            "    " * INDENT,
        ),
        file=OUTPUT_BUF,
    )
    INDENT += 1
    for bodyop in if_op.elseRegion.blocks[0].operations:
        if isinstance(bodyop, scf.YieldOp):
            print_yield_as_return(bodyop)
        else:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    INDENT -= 1


def generic_print_walk_callback(op):
    opview = op.opview
    if isinstance(opview, builtin.ModuleOp):
        for attr in opview.attributes:
            print(
                f"ctx.module.operation.attributes['{attr.name}'] = Attribute.parse('{(attr.attr)}')",
                file=OUTPUT_BUF,
            )
        return WalkResult.ADVANCE

    if isinstance(opview, func.FuncOp):
        print("", file=OUTPUT_BUF)
        print_func_op(opview)
    elif isinstance(opview, scf.ForOp):
        print_scf_for(opview)
    elif isinstance(opview, arith.ConstantOp):
        print_arith_constant(opview)
    elif isinstance(opview, scf.IfOp):
        print_scf_if(opview)
        return WalkResult.SKIP
    elif isinstance(opview, scf.YieldOp):
        print_opview(opview, name=f"scf.yield_")
    elif isinstance(opview, func.ReturnOp):
        print_opview(opview, name=f"func.return_")
    else:
        print_opview(opview)

    if len(op.regions):
        global INDENT
        INDENT += 1
        for bodyop in op.regions[0].blocks[0].operations:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        INDENT -= 1
        return WalkResult.SKIP

    return WalkResult.ADVANCE


def print_attr_alias(attr_line: str):
    print(attr_line)
    alias_name, attr_str = attr_line.split(" = ", maxsplit=1)
    assert alias_name.startswith("#")
    alias_name = alias_name[1:]
    attr = Attribute.parse(attr_str)
    print(f"{alias_name} = {map_attr(attr)}", file=OUTPUT_BUF)
    ATTR_ALIASES[attr] = alias_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()
    with mlir_mod_ctx(open(args.input_file).read()) as ctx:
        ctx.module.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)


if __name__ == "__main__":
    main()
