import inspect
import pkgutil
from types import SimpleNamespace
from typing import List

from mlir.dialects.transform.extras import OpHandle
from ...meta import region_op
from ...util import get_user_code_loc
from ....dialects import pdl
from ....dialects import transform
from ....dialects._ods_common import (
    _dispatch_mixed_values,
)
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects._structured_transform_ops_gen import (
    TileUsingForallOp,
    MatchOp,
)
from ....dialects.transform import *
from ....dialects.transform.structured import TileUsingForOp
from ....ir import Type, Operation, StringAttr, Attribute, Value

transform_fully_qualified_name = transform.__spec__.name


def create_simple_namespace(name):
    return SimpleNamespace(__name__=name)


# transform.apply_patterns is both a namespace and an op...
delattr(transform, "apply_patterns")

skips = {"_Dialect"}
for mod in pkgutil.iter_modules(transform.__path__):
    if mod.name.startswith("_") or mod.name == "extras":
        continue
    imported_module = __import__(
        f"{transform_fully_qualified_name}.{mod.name}", fromlist=["*"]
    )
    for name, obj in inspect.getmembers(imported_module):
        if (
            inspect.isclass(obj)
            and hasattr(obj, "OPERATION_NAME")
            and obj.__name__ not in skips
        ):
            if "_ops_gen" not in obj.__module__:
                obj = obj.__base__

            namespaces = obj.OPERATION_NAME.split(".")
            assert namespaces[0] == "transform"
            namespaces = namespaces[1:]
            value_builder_name = "_".join(namespaces)

            if namespaces[0] not in globals():
                globals()[namespaces[0]] = SimpleNamespace(__name__=namespaces[0])
            simple_namespace = globals()[namespaces[0]]

            for i, n in enumerate(namespaces[1:-1]):
                if not hasattr(simple_namespace, n):
                    # dumb: without the prefix, this somehow always names the modules "mlir.dialect.module.transform.<n>" instead of suffixing
                    setattr(
                        simple_namespace,
                        n,
                        SimpleNamespace(__name__=f"{simple_namespace.__name__}.{n}"),
                    )
                simple_namespace = getattr(simple_namespace, n)

            setattr(
                simple_namespace,
                namespaces[-1],
                getattr(imported_module, value_builder_name),
            )


pdl_operation_t = lambda: pdl.OperationType.get()


def _unwrap_op_handle(op_handle):
    assert isinstance(op_handle, OpHandle)
    return Value(op_handle)


def transform_any_op_t():
    return AnyOpType.get()


def transform_any_value_t():
    return AnyValueType.get()


def transform_op_t(name):
    return OperationType.get(name)


def sequence_(
    target: Optional[Union[Operation, Value, Type, str]] = None,
    target_tag=None,
    failure_propagation_mode: FailurePropagationMode = None,
    results_: List[Type] = None,
    extra_bindings: List[Value] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if results_ is None:
        results_ = []
    if target is None:
        target = pdl_operation_t()
    # this is a misnomer - it's not about targeting a particular op
    # but about picking which transform sequence runs using
    # transform_dialect_interpreter(debug_transform_root_tag="")
    if target_tag is None:
        target_tag = str(loc).split("/")[-1]
    if extra_bindings is None:
        extra_bindings = []
    if failure_propagation_mode is None:
        failure_propagation_mode = FailurePropagationMode.Propagate

    if isinstance(target, str):
        target = transform_op_t(target)

    seq_op = SequenceOp(
        failure_propagation_mode,
        results_,
        target,
        extra_bindings,  # loc=loc, ip=ip
    )
    seq_op.operation.attributes["transform.target_tag"] = StringAttr.get(target_tag)

    return seq_op


sequence = region_op(sequence_, terminator=YieldOp)

StrOrAttrList = Sequence[Union[StringAttr, str]]


def unroll(target: Value, factor=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return get_op_result_or_op_results(
        LoopUnrollOp(target, factor=factor, loc=loc, ip=ip)
    )


def match(
    target: Value,
    ops=None,
    *,
    interface=None,
    op_attrs=None,
    filter_result_type=None,
    matched_op=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if matched_op is None:
        matched_op = transform_any_op_t()
    return get_op_result_or_op_results(
        MatchOp(
            matched_op,
            target,
            ops=ops,
            interface=interface,
            op_attrs=op_attrs,
            filter_result_type=filter_result_type,
            loc=loc,
            ip=ip,
        )
    )


def tile_to_scf_for(
    target: Value,
    *,
    sizes: List[int],
    interchange=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    t = tuple(
        (
            get_op_result_or_op_results(
                TileUsingForOp(
                    target,
                    sizes=sizes,
                    interchange=interchange,
                    loc=loc,
                    ip=ip,
                )
            )
        )
    )

    return t[0], t[1:]


def tile_to_scf_forall(
    target,
    tile_sizes,
    num_threads=None,
    *,
    mapping=None,
    loc=None,
    ip=None,
):
    if num_threads is None:
        num_threads = []
    if loc is None:
        loc = get_user_code_loc()
    (
        dynamic_num_threads,
        packed_num_threads,
        static_num_threads,
    ) = _dispatch_mixed_values(num_threads)
    (
        dynamic_tile_sizes,
        packed_tile_sizes,
        static_tile_sizes,
    ) = _dispatch_mixed_values(tile_sizes)

    tiled_op = forall_op = target.type

    t = tuple(
        (
            get_op_result_or_op_results(
                TileUsingForallOp(
                    forall_op,
                    tiled_op,
                    target,
                    num_threads=dynamic_num_threads,
                    tile_sizes=dynamic_num_threads,
                    packed_num_threads=packed_num_threads,
                    packed_tile_sizes=packed_tile_sizes,
                    static_num_threads=static_num_threads,
                    static_tile_sizes=static_tile_sizes,
                    mapping=mapping,
                    loc=loc,
                    ip=ip,
                )
            )
        )
    )

    return t[0], t[1:]


__structured_fuse_into_containing_op = structured.fuse_into_containing_op


def _structured_fuse_into_containing_op(
    producer_op,
    containing_op,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __structured_fuse_into_containing_op(
        transform_any_op_t(),
        transform_any_op_t(),
        producer_op,
        containing_op,
        loc=loc,
        ip=ip,
    )


structured.fuse_into_containing_op = _structured_fuse_into_containing_op

_split_handle = split_handle


def split_handle(
    handle,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if not isinstance(handle.owner.opview, MatchOp):
        raise ValueError(f"{handle=} must be an instance of MatchOp")
    return _split_handle(
        [transform_any_op_t()] * len(list(handle.owner.opview.ops)),
        handle,
        loc=loc,
        ip=ip,
    )


__structured_pack = structured.pack


def _structured_pack(target, packed_sizes, *, packed_op=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    (
        dynamic_packed_sizes,
        # packed here means %1:2 packing (results packing)
        _packed_packed_sizes,
        static_packed_sizes,
    ) = _dispatch_mixed_values(packed_sizes)

    if packed_op is None:
        packed_op = transform_any_op_t()

    return __structured_pack(
        packed_op=packed_op,
        target=target,
        packed_sizes=dynamic_packed_sizes,
        static_packed_sizes=static_packed_sizes,
        loc=loc,
        ip=ip,
    )


structured.pack = _structured_pack

_get_producer_of_operand = get_producer_of_operand

get_producer_of_operand = lambda *args, **kwargs: _get_producer_of_operand(
    transform_any_op_t(), *args, **kwargs
)

__structured_pack_transpose = structured.pack_transpose


def _structured_pack_transpose(
    target_pack_or_un_pack_op,
    target_linalg_op,
    *,
    outer_perm=None,
    inner_perm=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __structured_pack_transpose(
        packed_op=transform_any_op_t(),
        pack_op=transform_any_op_t(),
        un_pack_op=transform_any_op_t(),
        target_pack_or_un_pack_op=target_pack_or_un_pack_op,
        target_linalg_op=target_linalg_op,
        outer_perm=outer_perm,
        inner_perm=inner_perm,
        loc=loc,
        ip=ip,
    )


structured.pack_transpose = _structured_pack_transpose


_include = include


def include(target, operands, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _include(
        results_=[],
        target=target,
        failure_propagation_mode=FailurePropagationMode.Propagate,
        operands_=operands,
        loc=loc,
        ip=ip,
    )


__structured_bufferize_to_allocation = structured.bufferize_to_allocation


def _structured_bufferize_to_allocation(
    target,
    *,
    memory_space=None,
    memcpy_op=None,
    alloc_op=None,
    bufferize_destination_only=None,
    emit_dealloc=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(memory_space, int):
        memory_space = str(memory_space)
    if isinstance(memory_space, str):
        try:
            memory_space = Attribute.parse(memory_space)
        except:
            memory_space = StringAttr.get(memory_space)

    return __structured_bufferize_to_allocation(
        allocated_buffer=transform_any_value_t(),
        new_ops=transform_any_op_t(),
        target=target,
        memory_space=memory_space,
        memcpy_op=memcpy_op,
        alloc_op=alloc_op,
        bufferize_destination_only=bufferize_destination_only,
        emit_dealloc=emit_dealloc,
        loc=loc,
        ip=ip,
    )


structured.bufferize_to_allocation = _structured_bufferize_to_allocation


_structured_lower_pack = structured.lower_pack
structured.lower_pack = lambda target: _structured_lower_pack(
    transform_op_t("tensor.pad"),
    transform_op_t("tensor.expand_shape"),
    transform_op_t("linalg.transpose"),
    target,
)


apply_patterns.canonicalization = transform.apply_patterns_canonicalization


_structured_vectorize_children_and_apply_patterns = (
    structured.vectorize_children_and_apply_patterns
)
structured.vectorize_children_and_apply_patterns = (
    lambda *args, **kwargs: _structured_vectorize_children_and_apply_patterns(
        transform_any_op_t(), *args, **kwargs
    )
)

_bufferization_one_shot_bufferize = bufferization.one_shot_bufferize
bufferization.one_shot_bufferize = (
    lambda *args, **kwargs: _bufferization_one_shot_bufferize(
        transform_any_op_t(), *args, **kwargs
    )
)
