from collections import OrderedDict
from dataclasses import dataclass

from sortedcontainers import SortedList

from ...ir import (
    Block,
    Value,
    Operation,
    OperationList,
    BlockArgument,
    OperationIterator,
)
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
        use_block = user.operation.block
        # Find an owner block in the current region. Note that a value does not
        # escape this block if it is used in a nested region.
        use_block = find_ancestor_block_in_region(use_block)
        if use_block != b:
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

        # Mark all block arguments (phis) as defined.
        for arg in block.arguments:
            self.def_values.add(arg)
            # how the fuck can block args escape a block?
            # answer:
            #   func.func @test(%arg0: i32, %arg1: i16) -> i16 {
            #     cf.br ^bb1(%arg1 : i16)
            #   ^bb1(%0: i16):  // pred: ^bb0
            #     cf.br ^bb2(%arg0 : i32)
            #   ^bb2(%1: i32):  // pred: ^bb1
            #     return %0 : i16
            #   }
            if escapes_block(arg, self.block):
                self.out_values.add(arg)

        # Gather out values of all operations in the current block.
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

    # Updates live-out information of the current block. It iterates over all
    # successors and unifies their live-in values with the current live-out
    # values.
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

    # Propagate the in and out-value sets (fixpoint iteration).
    while to_process:
        # Pairs are returned in LIFO order if last is true or FIFO order if false.
        current, _ = to_process.popitem(last=True)
        builder = builders[current]
        builder.update_liveout(builders)
        if not builder.update_livein():
            continue
        for p in current.predecessors:
            if p not in to_process:
                to_process[p] = True

    return builders


class LivenessBlockInfo:
    block: Block = None
    in_values: set[Value] = None
    out_values: set[Value] = None

    def __init__(self, block, in_values, out_values):
        self.block = block
        self.in_values = in_values
        self.out_values = out_values

    def is_livein(self, v: Value):
        return v in self.in_values

    def is_liveout(self, v: Value):
        return v in self.out_values

    def get_start_operation(self, v: Value):
        # The given value is either live-in or is defined
        # in the scope of this block.
        if self.is_livein(v) or isinstance(v, BlockArgument):
            return self.block.operations[0]
        return v.owner

    def get_end_operation(self, v: Value, start_op: Operation):
        # The given value is either dying in this block or live-out.
        if self.is_liveout(v):
            return self.block.operations[-1]
        # Resolve the last operation (must exist by definition).
        end_op = start_op
        for use in v.uses:
            # Find the associated operation in the current block (if any).
            # Check whether the use is in our block and after the current end
            # operation.
            if (
                use_op := find_ancestor_op_in_block(self.block, use.owner)
            ) and end_op.is_before_in_block(use_op):
                end_op = use_op
        return end_op

    def currently_live_values(self, op: Operation):
        live_set = set()

        # Given a value, check which ops are within its live range. For each of
        # those ops, add the value to the set of live values as-of that op.
        def add_value_to_currently_live_sets(value):
            # Determine the live range of this value inside this block.
            end_of_live_range = None
            # If it's a livein or a block argument, then the start is the beginning
            # of the block.
            if self.is_livein(value) or isinstance(value, BlockArgument):
                start_of_live_range = self.block.operations[0]
            else:
                start_of_live_range = find_ancestor_op_in_block(self.block, value.owner)

            # If it's a liveout, then the end is the back of the block.
            if self.is_liveout(value):
                end_of_live_range = self.block.operations[-1]

            # We must have at least a startOfLiveRange at this point. Given this, we
            # can use the existing getEndOperation to find the end of the live range.
            if start_of_live_range is not None and end_of_live_range is None:
                end_of_live_range = self.get_end_operation(value, start_of_live_range)

            assert end_of_live_range, "Must have end_of_live_range at this point!"
            # If this op is within the live range, insert the value into the set.
            if not (
                op.is_before_in_block(start_of_live_range)
                or end_of_live_range.is_before_in_block(op)
            ):
                live_set.add(value)

        for arg in self.block.arguments:
            add_value_to_currently_live_sets(arg)

        # Handle live-ins. Between the live ins and all the op results that gives us
        # every value in the block.
        for value in self.in_values:
            add_value_to_currently_live_sets(value)

        # Now walk the block and handle all values used in the block and values
        # defined by the block.
        for bop in self.block.operations:
            for r in bop.results:
                add_value_to_currently_live_sets(r)
            if bop == op:
                break

        return live_set


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

    def resolve_liveness(self, value: Value) -> OperationList:
        result = []
        to_process = OrderedDict()

        # Start with the defining block
        if isinstance(value, BlockArgument):
            current_block = value.owner
        else:
            current_block = value.owner.operation.block
        to_process[current_block] = True

        # Start with all associated blocks
        for use in value.uses:
            user = use.owner
            use_block = user.operation.block
            if use_block not in to_process:
                to_process[use_block] = True

        while to_process:
            current_block, _ = to_process.popitem(last=True)
            block_info = self.block_mapping[current_block]
            # Note that start and end will be in the same block.
            start = block_info.get_start_operation(value)
            end = block_info.get_end_operation(value, start)

            for op in OperationIterator(start.parent, start):
                if start == end:
                    break
                result.append(op)
            for succ in current_block.successors:
                if self.get_liveness(succ).is_livein(value) and succ not in to_process:
                    to_process[succ] = True

        return result

    def get_liveness(self, block: Block) -> LivenessBlockInfo | None:
        if liveness := self.block_mapping.get(block):
            return liveness

    def get_livein(self, block: Block) -> set[Value] | None:
        if liveness := self.get_liveness(block):
            return liveness.in_values

    def get_liveout(self, block: Block) -> set[Value] | None:
        if liveness := self.get_liveness(block):
            return liveness.out_values

    def is_dead_after(self, value: Value, op: Operation) -> bool:
        block_info = self.get_liveness(op.operation.block)
        if block_info.is_liveout(value):
            return False
        end_op = block_info.get_end_operation(value, op)
        # If the operation is a real user of `value` the first check is sufficient.
        # If not, we will have to test whether the end operation is executed before
        # the given operation in the block.
        return end_op == op or end_op.is_before_in_block(op)

    def __str__(self):
        print("// ---- Liveness -----")

        # Builds unique block/value mappings for testing purposes.
        block_ids: dict[Block, int] = {}
        operation_ids: dict[Operation, int] = {}
        value_ids: dict[Value, int] = {}
        for block in walk_blocks_in_operation(self.operation):
            block_ids[block] = len(block_ids)
            for argument in block.arguments:
                value_ids[argument] = len(value_ids)
            for operation in block.operations:
                operation_ids[operation] = len(operation_ids)
                for result in operation.results:
                    value_ids[result] = len(value_ids)

        # Local printing helpers
        def print_value_ref(value):
            if isinstance(value, BlockArgument):
                print(f"arg{value.arg_number}@{block_ids[value.owner]}", end=" ")
            else:
                print(f"val_{value_ids[value]}", end=" ")

        def print_value_refs(values: set[Value]):
            ordered_values = sorted(list(values), key=lambda v: value_ids[v])
            for value in ordered_values:
                print_value_ref(value)

        # Dump information about in and out values.
        for block in walk_blocks_in_operation(self.operation):
            print(f"// - Block: {block_ids[block]}")
            liveness = self.get_liveness(block)
            print("// --- LiveIn: ", end="")
            print_value_refs(liveness.in_values)
            print("\n// --- LiveOut: ", end="")
            print_value_refs(liveness.out_values)
            print()

            # Print liveness intervals.
            print("// --- BeginLivenessIntervals", end="")
            for op in block.operations:
                if not op.results:
                    continue
                print()
                for result in op.results:
                    print("// ", end="")
                    print_value_ref(result)
                    print(":", end="")
                    live_operations = sorted(
                        list(self.resolve_liveness(result)),
                        key=lambda v: operation_ids[v],
                    )
                    for operation in live_operations:
                        print("\n//     ", end="")
                        print(operation)
            print("\n// --- EndLivenessIntervals")

            # Print currently live values.
            print("// --- BeginCurrentlyLive")
            for op in block.operations:
                currently_live = liveness.currently_live_values(op)
                if not currently_live:
                    continue
                print("//     ", end="")
                print(op)
                print(" [", end="")
                print_value_refs(currently_live)
                print("\b]")
            print("// --- EndCurrentlyLive")

        print("// -------------------")


@dataclass(frozen=True)
class LiveInterval:
    start: int
    end: int
    name: str

    def __str__(self):
        return f"{self.name}@[{self.start},{self.end}]"

    def __repr__(self):
        return f"{self.name}@[{self.start},{self.end}]"


def linear_scan_register_allocation(intervals: list[LiveInterval], R: int):
    active: list[LiveInterval] = SortedList(key=lambda i: i.end)
    free_registers = set(range(R))
    register = {}
    location = {}

    # expire intervals whose lifetimes have ended
    # (remove from active and return the register to the free list)
    def expire_old_intervals(i: LiveInterval):
        for j in active:
            if j.end > i.start:
                return
            # j.end < i.start
            active.remove(j)
            free_registers.add(register[j])

    # spill either last ending
    # or this interval
    def spill_at_interval(i: LiveInterval):
        spill = active[-1]
        # if last ending ends after this interval
        if spill.end > i.end:
            # give its register to this interval
            register[i] = register[spill]
            # stack slot
            location[spill] = len(location)
            assert spill in active, "expected spill in active"
            active.remove(spill)
            active.add(i)
        else:
            # else this interval ends later so
            # spill it (give it a stack slot)
            location[i] = len(location)

    # sorted by start
    intervals = sorted(intervals, key=lambda i: i.start)
    for i in intervals:
        expire_old_intervals(i)
        # if max registers reached, spill
        if len(active) == R:
            spill_at_interval(i)
        else:
            register[i] = free_registers.pop()
            active.add(i)

    return register, location