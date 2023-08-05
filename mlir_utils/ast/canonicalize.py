import ast
import difflib
import enum
import inspect
import logging
import types
from abc import ABC
from opcode import opmap
from types import CodeType

import libcst as cst
from bytecode import ConcreteBytecode
from libcst._position import CodeRange, CodePosition
from libcst.matchers import MatcherDecoratableTransformer
from libcst.metadata import (
    PositionProvider,
    ParentNodeProvider,
    QualifiedNameProvider,
    ExperimentalReentrantCodegenProvider,
)

from mlir_utils.ast.util import get_module_cst, copy_func

logger = logging.getLogger(__name__)


class Transformer(MatcherDecoratableTransformer):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
        QualifiedNameProvider,
        ExperimentalReentrantCodegenProvider,
    )

    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno

    def get_pos(self, node):
        pos = self.get_metadata(PositionProvider, node)
        return CodeRange(
            CodePosition(pos.start.line + self.first_lineno, pos.start.column),
            CodePosition(pos.end.line + self.first_lineno, pos.end.column),
        )

    def get_parent(self, node):
        # NB: can only call this on "original nodes"
        return self.get_metadata(ParentNodeProvider, node)

    def get_code_snippet(self, node):
        return self.get_metadata(
            ExperimentalReentrantCodegenProvider, node
        ).get_original_statement_code()


class StrictTransformer(Transformer):
    def visit_FunctionDef(self, node: cst.FunctionDef):
        return False


def transform_func(f, *transformer_ctors):
    module_cst = get_module_cst(f)
    context = types.SimpleNamespace()
    for transformer_ctor in transformer_ctors:
        orig_code = module_cst.code
        wrapper = cst.MetadataWrapper(module_cst)
        func_node = wrapper.module.body[0]
        replace = transformer_ctor(
            context=context, first_lineno=f.__code__.co_firstlineno - 1
        )
        logger.debug("[transformer] %s", replace.__class__.__name__)
        with replace.resolve(wrapper):
            new_func = func_node._visit_and_replace_children(replace)
        module_cst = wrapper.module.deep_replace(func_node, new_func)
        new_code = module_cst.code

        diff = list(
            difflib.unified_diff(
                orig_code.splitlines(),  # to this
                new_code.splitlines(),  # delta from this
                lineterm="",
            )
        )
        logger.debug("[transformed code diff]\n\n%s", "\n" + "\n".join(diff))
    logger.debug("[final transformed code]\n\n%s", module_cst.code)

    return module_cst


def transform_cst(
    f, transformers: list[type(Transformer) | type(StrictTransformer)] = None
):
    if transformers is None:
        return f

    module_cst = transform_func(f, *transformers)

    code = "\n" * (f.__code__.co_firstlineno - 1) + module_cst.code
    module_code_o = compile(code, f.__code__.co_filename, "exec")
    new_f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == f.__name__
    )

    return copy_func(f, new_f_code_o)


# this is like this because i couldn't figure out how to subclass
# Enum and simultaneously pass in opmap
OpCode = enum.Enum("OpCode", opmap)


def to_int(self: OpCode):
    return self.value


def to_str(self: OpCode):
    return self.name


setattr(OpCode, "__int__", to_int)
setattr(OpCode, "__str__", to_str)


class BytecodePatcher(ABC):
    def __init__(self, context=None):
        self.context = context

    @property
    def patch_bytecode(self, code: ConcreteBytecode, original_f) -> ConcreteBytecode:
        pass


def patch_bytecode(f, patchers: list[type(BytecodePatcher)] = None):
    if patchers is None:
        return f
    code = ConcreteBytecode.from_code(f.__code__)
    context = types.SimpleNamespace()
    for patcher in patchers:
        code = patcher(context).patch_bytecode(code, f)

    return copy_func(f, code.to_code())


class Canonicalizer(ABC):
    @property
    def cst_transformers(self) -> list[StrictTransformer]:
        pass

    @property
    def bytecode_patchers(self) -> list[BytecodePatcher]:
        pass


def canonicalize(*, using: Canonicalizer):
    def wrapper(f):
        f = transform_cst(f, using.cst_transformers)
        f = patch_bytecode(f, using.bytecode_patchers)
        return f

    return wrapper
