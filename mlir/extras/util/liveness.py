from collections import deque

from ...ir import Block, Value, Operation
from .util import (
    walk_blocks_in_operation,
    walk_operations,
    find_ancestor_block_in_region,
)


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

        def gather_out_values(v: Value):
            for use in v.uses:
                user = use.owner.operation
                owner_block = user.block
                owner_block = find_ancestor_block_in_region(owner_block)
                if owner_block != block:
                    self.out_values.add(v)
                    break

        for arg in block.arguments:
            gather_out_values(arg)
        for op in block.operations:
            for r in op.results:
                gather_out_values(r)

        for op in block.operations:
            for nested_op in walk_operations(op):
                self.def_values |= set(nested_op.results)
                self.use_values |= set(nested_op.operands)
                for b in walk_blocks_in_operation(nested_op):
                    self.def_values |= set(b.arguments)

        self.use_values -= self.def_values

    def update_livein(self):
        new_in = self.use_values
        new_in |= self.out_values
        new_in -= self.def_values

        if len(new_in) == len(self.in_values):
            return set()
        self.in_values = new_in
        return new_in

    def update_liveout(self, builders):
        for succ in self.block.successors:
            self.out_values -= builders[succ].in_values


def build_block_mapping(op):
    visited = set()
    to_process = deque()
    builders = {}
    for b in walk_blocks_in_operation(op):
        builder = builders[b] = BlockInfoBuilder(b)
        if builder.update_livein():
            for p in b.predecessors:
                if p not in visited:
                    to_process.append(p)
                    visited.add(p)

    while to_process:
        current = to_process.popleft()
        builder = builders[current]
        builder.update_liveout(builders)
        if builder.update_livein():
            for p in current.predecessors:
                if p not in visited:
                    to_process.append(p)
                    visited.add(p)

    return builders


class LivenessBlockInfo:
    block = None
    in_values = None
    out_values = None

    def __init__(self, block, in_values, out_values):
        self.block = block
        self.in_values = in_values
        self.out_values = out_values


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
