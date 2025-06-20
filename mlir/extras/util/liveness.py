from collections import OrderedDict

from ...ir import Block, Value, Operation
from .util import (
    walk_blocks_in_operation,
    walk_operations,
    find_ancestor_block_in_region,
    find_ancestor_op_in_block,
)

# based on https://github.com/llvm/llvm-project/blob/07ae19c132e1b0adbdb3cc036b9f50624e2ed1b7/mlir/lib/Analysis/Liveness.cpp


def escapes_block(v: Value, b: Block):
    # Check if value escapes, i.e., if there's a use
    # which is in a block that is not "our" block
    for use in v.uses:
        user = use.owner
        owner_block = user.block
        owner_block = find_ancestor_block_in_region(owner_block)
        if owner_block != b:
            return True
    return False


class BlockInfoBuilder:
    block = None
    in_values = None
    out_values = None
    def_values = None
    use_values = None

    def __init__(self, block):
        self.block = block
        self.in_values = set()
        self.out_values = set()
        self.def_values = set()
        self.use_values = set()

        for arg in block.arguments:
            self.def_values.add(arg)
            if escapes_block(arg, self.block):
                self.out_values.add(arg)
        for op in block.operations:
            for r in op.results:
                if escapes_block(r, self.block):
                    self.out_values.add(r)

        # Mark all nested operation results as defined, and nested operation
        # operands as used. All defined value will be removed from the used set
        # at the end.
        for op in block.operations:
            for nested_op in walk_operations(op):
                self.def_values |= set(nested_op.results)
                self.use_values |= set(nested_op.operands)
                for b in walk_blocks_in_operation(nested_op):
                    self.def_values |= set(b.arguments)

        self.use_values -= self.def_values

    # newIn = use U out - def
    def update_livein(self):
        new_in = (self.use_values | self.out_values) - self.def_values
        # It is sufficient to check the set sizes (instead of their contents) since
        # the live-in set can only grow monotonically during all update operations.
        if len(new_in) == len(self.in_values):
            return set()
        self.in_values = new_in
        return new_in

    def update_liveout(self, builders):
        for succ in self.block.successors:
            self.out_values |= builders[succ].in_values


def build_block_mapping(op) -> dict[Block, BlockInfoBuilder]:
    to_process = OrderedDict()
    builders = {}
    for b in walk_blocks_in_operation(op):
        builder = builders[b] = BlockInfoBuilder(b)
        if not builder.update_livein():
            continue
        for p in b.predecessors:
            if p in to_process:
                continue
            to_process[p] = True

    while to_process:
        # Pairs are returned in LIFO order if last is true or FIFO order if false.
        current, _ = to_process.popitem(last=False)
        builder = builders[current]
        builder.update_liveout(builders)
        if not builder.update_livein():
            continue
        for p in current.predecessors:
            if p in to_process:
                continue
            to_process[p] = True

    return builders


class LivenessBlockInfo:
    block: Block = None
    in_values = None
    out_values = None

    def __init__(self, block, in_values, out_values):
        self.block = block
        self.in_values = in_values
        self.out_values = out_values

    def is_livein(self, v: Value):
        return v in self.in_values

    def is_liveout(self, v: Value):
        return v in self.out_values

    def get_start_operation(self, v: Value):
        owner = v.owner
        if self.is_livein(v) or not owner:
            return self.block.operations[0]
        return owner

    def get_end_operation(self, v: Value, start_op: Operation):
        if self.is_livein(v):
            return self.block.operations[-1]
        end_op = start_op
        for use in v.uses:
            use_op = find_ancestor_op_in_block(use.owner)


class Liveness:
    operation = None
    block_mapping = None

    def __init__(self, op):
        self.operation = op
        self.block_mapping = {}

        builders = build_block_mapping(self.operation)
        for block, builder in builders.items():
            assert block == builder.block
            self.block_mapping[block] = LivenessBlockInfo(
                builder.block, builder.in_values, builder.out_values
            )
