import ast
import difflib
import enum
import inspect
import logging
import types
import warnings
from abc import ABC, abstractmethod
from dis import findlinestarts
from opcode import opmap
from types import CodeType
from typing import List, Union, Sequence

import astunparse
from bytecode import ConcreteBytecode

from ..ast.util import get_module_cst, set_lineno

logger = logging.getLogger(__name__)


class Transformer(ast.NodeTransformer):
    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno


class StrictTransformer(Transformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node


def transform_func(f, *transformer_ctors: type(Transformer)):
    module = get_module_cst(f)
    context = types.SimpleNamespace()
    for transformer_ctor in transformer_ctors:
        orig_code = astunparse.unparse(module)
        func_node = module.body[0]
        replace = transformer_ctor(
            context=context, first_lineno=f.__code__.co_firstlineno - 1
        )
        logger.debug("[transformer] %s", replace.__class__.__name__)
        func_node = replace.generic_visit(func_node)
        new_code = astunparse.unparse(func_node)

        diff = list(
            difflib.unified_diff(
                orig_code.splitlines(),  # to this
                new_code.splitlines(),  # delta from this
                lineterm="",
            )
        )
        logger.debug("[transformed code diff]\n%s", "\n" + "\n".join(diff))
        logger.debug("[transformed code]\n%s", new_code)
        module.body[0] = func_node

    logger.debug("[final transformed code]\n\n%s", new_code)

    return module


# TODO(max): unify with `replace_closure` in ast/utils.py
def insert_closed_vars(f, module):
    enclosing_mod = ast.FunctionDef(
        name="enclosing_mod",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[],
        decorator_list=[],
        type_params=[],
    )
    for var in f.__code__.co_freevars:
        enclosing_mod.body.append(
            ast.Assign(
                targets=[ast.Name(var, ctx=ast.Store())],
                value=ast.Constant(None, kind="None"),
            )
        )
    enclosing_mod = set_lineno(enclosing_mod, module.body[0].lineno)
    enclosing_mod = ast.fix_missing_locations(enclosing_mod)

    enclosing_mod.body.extend(module.body)
    module.body = [enclosing_mod]
    return module


def find_func_in_code_object(co, func_name):
    for c in co.co_consts:
        if type(c) is CodeType:
            if c.co_name == func_name:
                return c
            else:
                f = find_func_in_code_object(c, func_name)
                if f is not None:
                    return f


def transform_ast(
    f, transformers: List[Union[type(Transformer), type(StrictTransformer)]] = None
):
    if transformers is None:
        return f

    module = transform_func(f, *transformers)
    if f.__closure__:
        module = insert_closed_vars(f, module)
    module = ast.fix_missing_locations(module)
    module = ast.increment_lineno(module, f.__code__.co_firstlineno - 1)
    module_code_o = compile(module, f.__code__.co_filename, "exec")
    new_f_code_o = find_func_in_code_object(module_code_o, f.__name__)
    n_lines = len(inspect.getsource(f).splitlines())
    line_starts = list(findlinestarts(new_f_code_o))
    if (
        max([l for _, l in line_starts]) - min([l for _, l in line_starts]) + 1
        > n_lines
    ) or (f.__code__.co_firstlineno != min([l for _, l in line_starts])):
        logger.debug(
            "something went wrong with the line numbers for the rewritten/canonicalized function"
        )
    f.__code__ = new_f_code_o
    return f


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

    @abstractmethod
    def patch_bytecode(self, code: ConcreteBytecode, original_f) -> ConcreteBytecode:
        pass


def patch_bytecode(f, patchers: List[type(BytecodePatcher)] = None):
    if patchers is None:
        return f
    code = ConcreteBytecode.from_code(f.__code__)
    context = types.SimpleNamespace()
    for patcher in patchers:
        code = patcher(context).patch_bytecode(code, f)

    f.__code__ = code.to_code()
    return f


class Canonicalizer(ABC):
    @property
    @abstractmethod
    def cst_transformers(self) -> List[StrictTransformer]:
        pass

    @property
    @abstractmethod
    def bytecode_patchers(self) -> List[BytecodePatcher]:
        pass


def canonicalize(*, using: Union[Canonicalizer, Sequence[Canonicalizer]]):
    if not isinstance(using, Sequence):
        using = [using]
    cst_transformers = []
    bytecode_patchers = []
    for u in using:
        cst_transformers.extend(u.cst_transformers)
        bytecode_patchers.extend(u.bytecode_patchers)

    def wrapper(f):
        f = transform_ast(f, cst_transformers)
        f = patch_bytecode(f, bytecode_patchers)
        return f

    return wrapper
