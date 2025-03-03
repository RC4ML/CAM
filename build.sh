#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OUTPUT_DIR = ../../build/lib




export LD_LIBRARY_PATH=${SCRIPT_DIR}/build/lib:${SCRIPT_DIR}/spdk/build/lib:$LD_LIBRARY_PATH
cd src/CAM_lib
make
cd ../CAM_variable_core_lib
make
# nvcc -rdc=true sample_read.cu ${OUTPUT_DIR}/gpu_transfer.o -o ${OUTPUT_DIR}/sample_read   -I../../src/GPU_memory_lib -L ${OUTPUT_DIR} -lCAM_interface
# nvcc -rdc=true sample_write.cu ${OUTPUT_DIR}/gpu_transfer.o -o ${OUTPUT_DIR}/sample_write  -I../../src/GPU_memory_lib -L ${OUTPUT_DIR} -lCAM_interface
cd ../benchmarks/CAM_benchmark
make clean
make
cd ../CAM_variable_core_benchmark
make clean
make
cd ../../applications/gemm
make