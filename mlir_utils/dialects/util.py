from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values


def get_result_or_results(op):
    return (
        get_op_results_or_values(op)
        if len(op.operation.results) > 1
        else get_op_result_or_value(op)
        if len(op.operation.results) > 0
        else None
    )
