NVCC = nvcc
INCLUDE = -I./include
FLAGS = -D__DEBUG__ -rdc=true -g -lcublas
OUTPUT_DIR = ../../../build/application/gemm

# Ensure the output directory exists
$(shell mkdir -p $(OUTPUT_DIR))


gemm-test: src/queue.cu src/device.cu src/controller.cu src/controller_decouple.cu src/controller_legacy.cu gemm/cam_gemm.cu
	$(NVCC) $(INCLUDE) $(FLAGS) -I /home/szy/yzh_hyprion/spdk_interface  -L /home/szy/yzh_hyprion/spdk_interface -lgpussd_baseline $^ -o $(OUTPUT_DIR)/$@




clean:
	rm -f gemm-test 