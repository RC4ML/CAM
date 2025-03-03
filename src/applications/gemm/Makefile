NVCC = nvcc
INCLUDE = -I./include
FLAGS = -D__DEBUG__ -rdc=true -g -lcublas
OUTPUT_DIR = ../../../build/application/gemm
LIB_PATH = ../../../build/lib
# Ensure the output directory exists
$(shell mkdir -p $(OUTPUT_DIR))


gemm-test: src/queue.cu src/device.cu src/controller.cu src/controller_decouple.cu src/controller_legacy.cu gemm/cam_gemm.cu
	$(NVCC) $(INCLUDE) $(FLAGS) -I../../GPU_memory_lib -I ../../../src/CAM_lib  -L../../../spdk/build/lib  -L ../../../build/lib -lCAM_interface    $^ -o $(OUTPUT_DIR)/$@




clean:
	rm -f gemm-test 