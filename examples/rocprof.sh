#!/bin/bash

#set -eux

export PATH=/opt/rocm-6.5.0/bin:$PATH
export PYTHONPATH=/home/mlevental/dev_projects/mlir-python-extras
export OUTPUT_PATH=$PWD
export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm-6.5.0/att-decoder-v3-3.0.0-Linux/lib

rm -rf traces
#rocprofv2 --kernel-trace /home/mlevental/dev_projects/mlir-python-extras/examples/demo.py
#rocprofv2 -i att.txt --kernel-trace --plugin att auto --mode file,csv -d traces/ /home/mlevental/dev_projects/mlir-python-extras/examples/demo.py
/opt/rocm-6.5.0/bin/rocprofv3 -i att.json -d traces -- /home/mlevental/dev_projects/mlir-python-extras/examples/demo.py
../../ROCProfiler-ATT-Viewer-amd-staging/cmake-build-debug/ATTViewer traces/ui*