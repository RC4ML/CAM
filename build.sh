cd src/CAM_lib
make
cd ../CAM_variable_core_lib
make
cd ../benchmarks/CAM_benchmark
make clean
make
cd ../CAM_variable_core_benchmark
make clean
make
cd ../../applications/gemm
make