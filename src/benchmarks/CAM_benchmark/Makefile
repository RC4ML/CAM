# Define variables for include, library, and output paths
INCLUDE_PATH = ../../CAM_lib
LIB_PATH = ../../../build/lib
OUTPUT_DIR = ../../../build/benchmarks/CAM_benchmark

# Ensure the output directory exists
$(shell mkdir -p $(OUTPUT_DIR))

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -I $(INCLUDE_PATH) -I../../GPU_memory_lib -L $(LIB_PATH) -L../../../spdk/build/lib -lspdk_interface 



TARGET1 = test_seq_read
SOURCE1 = test_seq_read.cu

TARGET2 = test_seq_write
SOURCE2 = test_seq_write.cu

TARGET3 = test_random_read
SOURCE3 = test_random_read.cu

TARGET4 = test_random_write
SOURCE4 = test_random_write.cu

# Default rule
all: $(OUTPUT_DIR)/$(TARGET1) $(OUTPUT_DIR)/$(TARGET2) $(OUTPUT_DIR)/$(TARGET3) $(OUTPUT_DIR)/$(TARGET4)

$(OUTPUT_DIR)/$(TARGET1): $(SOURCE1)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<

$(OUTPUT_DIR)/$(TARGET2): $(SOURCE2)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<

$(OUTPUT_DIR)/$(TARGET3): $(SOURCE3)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<

$(OUTPUT_DIR)/$(TARGET4): $(SOURCE4)
	$(NVCC) -o $@ $(NVCC_FLAGS) $<

# Clean rule
clean:
	rm -f $(OUTPUT_DIR)/$(TARGET1) $(OUTPUT_DIR)/$(TARGET2) $(OUTPUT_DIR)/$(TARGET3) $(OUTPUT_DIR)/$(TARGET4)