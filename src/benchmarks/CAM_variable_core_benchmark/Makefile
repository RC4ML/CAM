# Define variables for include, library, and output paths
INCLUDE_PATH = ../../CAM_variable_core_lib
LIB_PATH = ../../../build/lib
OUTPUT_DIR = ../../../build/benchmarks/CAM_variable_core_benchmark

# Ensure the output directory exists
$(shell mkdir -p $(OUTPUT_DIR))

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -I $(INCLUDE_PATH) -I../../GPU_memory_lib -L $(LIB_PATH) -lCAM_variable_core




TARGET1 = variable_core_test_read
SOURCE1 = variable_core_test_read.cu

TARGET2 = variable_core_test_write
SOURCE2 = variable_core_test_write.cu



# Default rule
all: $(OUTPUT_DIR)/$(TARGET1) $(OUTPUT_DIR)/$(TARGET2)

$(OUTPUT_DIR)/$(TARGET1): $(SOURCE1)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<

$(OUTPUT_DIR)/$(TARGET2): $(SOURCE2)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<



# Clean rule
clean:
	rm -f $(OUTPUT_DIR)/$(TARGET1) $(OUTPUT_DIR)/$(TARGET2)