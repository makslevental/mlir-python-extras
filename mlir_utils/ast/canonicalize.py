import ast
import inspect
import types
from abc import ABC
from types import CodeType
from typing import Type, Optional

import libcst as cst
from bytecode import ConcreteBytecode
from libcst.matchers import MatcherDecoratableTransformer
from libcst.metadata import ParentNodeProvider

from mlir_utils.ast.util import get_module_cst, copy_func


class FuncIdentTypeTable(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, f):
        super().__init__()
        self.ident_type: dict[str, Type] = {}
        module_cst = get_module_cst(f)
        wrapper = cst.MetadataWrapper(module_cst)
        wrapper.visit(self)

    def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
        parent = self.get_metadata(ParentNodeProvider, node)
        if isinstance(node.annotation, (cst.Tuple, cst.List)):
            self.ident_type[parent.target.value] = [
                e.value.value for e in node.annotation.elements
            ]
        else:
            self.ident_type[parent.target.value] = [node.annotation.value]

    def __getitem__(self, ident):
        return self.ident_type[ident]


class StrictTransformer(MatcherDecoratableTransformer):
    def __init__(self, context, func_sym_table: FuncIdentTypeTable):
        super().__init__()
        self.context = context
        self.func_sym_table = func_sym_table

    def visit_FunctionDef(self, node: cst.FunctionDef):
        return False


def transform_cst(f, transformers: list[type(StrictTransformer)] = None):
    if transformers is None:
        return f

    module_cst = get_module_cst(f)
    func_sym_table = FuncIdentTypeTable(f)
    context = types.SimpleNamespace()
    for transformer in transformers:
        func_node = module_cst.body[0]
        replace = transformer(context, func_sym_table)
        new_func = func_node._visit_and_replace_children(replace)
        module_cst = module_cst.deep_replace(func_node, new_func)

    tree = ast.parse(module_cst.code, filename=inspect.getfile(f))
    tree = ast.increment_lineno(tree, f.__code__.co_firstlineno - 1)
    module_code_o = compile(tree, f.__code__.co_filename, "exec")
    new_f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == f.__name__
    )

    return copy_func(f, new_f_code_o)


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


def canonicalize(*, with_: Canonicalizer):
    def wrapper(f):
        f = transform_cst(f, with_.cst_transformers)
        f = patch_bytecode(f, with_.bytecode_patchers)
        return f

    return wrapper
