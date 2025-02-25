NVCC = nvcc

CUDAFLAGS = -arch=sm_80 -Xcompiler=-fPIC,-mavx512f
CUDALDFLAGS = -lgdrapi -lcuda -lcudart 

# Output directory
OUTPUT_DIR = ../../build/lib

# Source and object files
SOURCE1 = GPU_memory_management.cpp
OBJECT1 = $(OUTPUT_DIR)/$(SOURCE1:.cpp=.o)
TARGET1 = $(OUTPUT_DIR)/libgpu_memory_management.so

# Default rule
all: $(TARGET1)

# Rule to compile the source file into an object file
$(OBJECT1): $(SOURCE1)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Rule to link the object file and create the shared library
$(TARGET1): $(OBJECT1)
	$(NVCC) -shared $(OBJECT1) -o $@ $(CUDALDFLAGS)

# Clean rule to remove generated files
clean:
	rm -f $(OBJECT1) $(TARGET1)