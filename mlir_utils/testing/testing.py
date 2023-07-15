import shutil
import tempfile
from subprocess import Popen, PIPE
from textwrap import dedent

import pytest

from mlir_utils.context import MLIRContext, mlir_mod_ctx
from .generate_test_checks import main


def filecheck(correct: str, module):
    filecheck_path = shutil.which("FileCheck")
    assert filecheck_path is not None, "couldn't find FileCheck"

    correct = dedent(correct)
    op = dedent(str(module).strip())
    with tempfile.NamedTemporaryFile() as tmp:
        correct_with_checks = main(correct)
        tmp.write(correct_with_checks.encode())
        tmp.flush()
        p = Popen([filecheck_path, tmp.name], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out, err = map(lambda o: o.decode(), p.communicate(input=op.encode()))
        if len(err):
            raise ValueError(err)


@pytest.fixture
def mlir_ctx() -> MLIRContext:
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        yield ctx
