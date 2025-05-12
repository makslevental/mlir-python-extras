#!/bin/bash

#set -eux

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

export PATH=/opt/rocm-6.5.0/bin:$PATH
export PYTHONPATH=$SCRIPT_DIR/..
export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm-6.5.0/att-decoder-v3-3.0.0-Linux/lib
export ATT_VIEWER=../../ROCProfiler-ATT-Viewer-amd-staging/cmake-build-debug/ATTViewer


rm -rf traces
rocprofv3 -i att.json -d traces -o demo_trace -- $SCRIPT_DIR/schedule_barriers.py

for ui in $(ls $SCRIPT_DIR/traces) ; do
  if [ -d $SCRIPT_DIR/traces/$ui ]; then
    ls $SCRIPT_DIR/traces/$ui | grep se > /dev/null
    if [ $? == 0 ]; then
      UI_PATH=$SCRIPT_DIR/traces/$ui
    fi
  fi
done

$ATT_VIEWER $UI_PATH