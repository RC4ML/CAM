SPDK_DIR = ../../spdk

CUDA_DIR = /usr/local/cuda-12.4
OUTPUT_DIR = ../../build/lib


$(shell mkdir -p $(OUTPUT_DIR))
	
.PHONY:all

exe:
	g++ --std=c++17 -mcmodel=medium CAM_interface.h CAM_interface.cpp  -o $(OUTPUT_DIR)/libCAM_interface.so -shared -lcudart \
	-L/usr/local/cuda/lib64 -L../../build/lib -lgpu_memory_management -lcudart -lcuda -L$(CUDA_DIR)/lib64 -I../../src/GPU_memory_lib \
	-g -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fno-strict-aliasing \
	-I$(SPDK_DIR)/include -march=native -Wformat -Wformat-security -D_GNU_SOURCE -fPIC -fstack-protector -fno-common \
	-I$(SPDK_DIR)/isa-l/.. -I$(SPDK_DIR)/isalbuild -I$(SPDK_DIR)/isa-l-crypto/.. -I$(CUDA_DIR)/include \
	-I$(SPDK_DIR)/isalcryptobuild -DNDEBUG -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -DSPDK_GIT_COMMIT=6ee9cd257 -pthread -std=c++17 \
	-Wl,-z,relro,-z,now -Wl,-z,noexecstack -fuse-ld=bfd -Wl,-rpath=$(SPDK_DIR)/isa-l/.libs -Wl,-rpath=$(SPDK_DIR)/isa-l-crypto/.libs \
	-L$(SPDK_DIR)/build/lib -Wl,--whole-archive -Wl,--no-as-needed -lspdk_sock_posix -lspdk_nvme -lspdk_keyring -lspdk_sock -lspdk_trace -lspdk_rpc -lspdk_jsonrpc -lspdk_json -lspdk_dma -lspdk_vmd -lspdk_util -lspdk_log \
	-Wl,--no-whole-archive $(SPDK_DIR)/build/lib/libspdk_env_dpdk.so -Wl,--no-as-needed $(SPDK_DIR)/dpdk/build/lib/librte_bus_pci.so $(SPDK_DIR)/dpdk/build/lib/librte_cryptodev.so \
	$(SPDK_DIR)/dpdk/build/lib/librte_dmadev.so $(SPDK_DIR)/dpdk/build/lib/librte_eal.so $(SPDK_DIR)/dpdk/build/lib/librte_ethdev.so \
	$(SPDK_DIR)/dpdk/build/lib/librte_hash.so $(SPDK_DIR)/dpdk/build/lib/librte_kvargs.so $(SPDK_DIR)/dpdk/build/lib/librte_log.so $(SPDK_DIR)/dpdk/build/lib/librte_mbuf.so \
	$(SPDK_DIR)/dpdk/build/lib/librte_mempool.so $(SPDK_DIR)/dpdk/build/lib/librte_mempool_ring.so $(SPDK_DIR)/dpdk/build/lib/librte_net.so $(SPDK_DIR)/dpdk/build/lib/librte_pci.so \
	$(SPDK_DIR)/dpdk/build/lib/librte_power.so $(SPDK_DIR)/dpdk/build/lib/librte_rcu.so $(SPDK_DIR)/dpdk/build/lib/librte_ring.so $(SPDK_DIR)/dpdk/build/lib/librte_telemetry.so \
	$(SPDK_DIR)/dpdk/build/lib/librte_vhost.so -Wl,-as-needed -Wl,-rpath=$(SPDK_DIR)/dpdk/build/lib  -L$(SPDK_DIR)/isa-l/.libs -lisal -L$(SPDK_DIR)/isa-l-crypto/.libs -lisal_crypto -pthread -lrt -luuid -lssl -lcrypto -lm -lfuse3 -lkeyutils -laio

	
	nvcc -dc   -Xcompiler -fPIC -I../../src/GPU_memory_lib -c gpu_transfer.cu -o $(OUTPUT_DIR)/gpu_transfer.o
	nvcc -dlink  -I ../../src/CAM_lib -o $(OUTPUT_DIR)/gpu_transfer_link.o $(OUTPUT_DIR)/gpu_transfer.o
	nvcc -rdc=true sample_read.cu $(OUTPUT_DIR)/gpu_transfer.o -o $(OUTPUT_DIR)/sample_read   -I../../src/GPU_memory_lib -L $(OUTPUT_DIR) -lCAM_interface
	nvcc -rdc=true sample_write.cu $(OUTPUT_DIR)/gpu_transfer.o -o $(OUTPUT_DIR)/sample_write  -I../../src/GPU_memory_lib -L $(OUTPUT_DIR) -lCAM_interface



