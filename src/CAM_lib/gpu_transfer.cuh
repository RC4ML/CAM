#ifndef __GPU_TRANSFER_CUH__
#define __GPU_TRANSFER_CUH__
#include <cuda_runtime.h>
#include <thread>
#include "CAM_interface.h"

#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__

#define MAX_EMBED_NUM  1000000



//* A semaphore for sending signals from the host to the device.
class Host2DeviceSemaphore  {
 private:
    void* InboundSemaphore;
    u_int64_t expectedInboundSemaphore;
    void* outboundSemaphore;
    u_int64_t* p_outboundSemaphoreValue;
    cudaStream_t stream;
    u_int64_t* total_num;
 public:
    Host2DeviceSemaphore(void){ 
        cudaHostAlloc( (void**)&p_outboundSemaphoreValue,sizeof(u_int64_t),cudaHostAllocDefault ) ;
        expectedInboundSemaphore =1; 
        cudaMalloc(&outboundSemaphore,sizeof(u_int64_t));
        *p_outboundSemaphoreValue =0;
       
    }
    void ConnectToDeviceSemaphore(void* InboundSemaphore_,u_int64_t* total_num_ ) {InboundSemaphore=InboundSemaphore_;total_num=total_num_;}
    void* GetoutboundSemaphore(void) { return outboundSemaphore;}
    u_int64_t GetTotalNumber(void) { return *total_num;}
    void ConnectToStream(cudaStream_t stream1){ stream= stream1;}
    void signal() {
        //printf("signal\n");
        (*p_outboundSemaphoreValue)+=1;  
        cudaError_t return_value=cudaMemcpyAsync(outboundSemaphore, p_outboundSemaphoreValue, sizeof(u_int64_t), cudaMemcpyHostToDevice,stream);
        if (return_value != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorName(return_value) << " - " << cudaGetErrorString(return_value) << std::endl;
            // 处理错误
        } 
        
        //printf("p_outboundSemaphoreValue: %ld\n",*(u_int64_t*)p_outboundSemaphoreValue);
    }

    void wait() {
        // printf("wait\n");
        // printf("InboundSemaphore: %ld\n",*(u_int64_t*)InboundSemaphore);
        // printf("expectedInboundSemaphore: %ld\n",expectedInboundSemaphore);
        uint64_t start = 0;
        while((*(u_int64_t*)InboundSemaphore) < expectedInboundSemaphore){
            start = 0;
            while (start++ < 100000);
            
        }
        // printf("wait end\n");
        // printf("InboundSemaphore: %ld\n",*(u_int64_t*)InboundSemaphore);
        // printf("expectedInboundSemaphore: %ld\n",expectedInboundSemaphore);
        expectedInboundSemaphore ++;
    }
};

// struct SmDevice2DeviceSemaphoreDeviceHandle {
    




//     MSCCLPP_DEVICE_INLINE void signal(uint64_t num) {
//         *total_num = num;
//         semaphoreIncrement();
//         *outboundSemaphoreId = *outboundSemaphoreValue;
//     }

//     /// Increase the counter of the local semaphore.
//     MSCCLPP_DEVICE_INLINE void semaphoreIncrement() { *outboundSemaphoreValue += 1; }

//     /// Get the value of the local semaphore.
//     MSCCLPP_DEVICE_INLINE uint64_t semaphoreGetLocal() const { return *outboundSemaphoreValue; }

    
// };




void SemaphoreInit(cudaStream_t stream1);
void Init(u_int32_t access_size,cudaStream_t stream1);
extern "C" __global__ void init_myKernel(void);

//* read functions
void polling_thread(void);
__device__ void prefetch(int64_t embed_num,uintptr_t *dev_addr);
__device__ void prefetch_syncronize(void);
uint64_t* get_d_data(void);
//*wrtie functions
void polling_thread_write(void);
__device__ void writeback(int64_t embed_num,uintptr_t *dev_addr);
__device__ void writeback_syncronize(void);
uint64_t* get_d_data_write(void);

void polling_thread_seq(void);
void polling_thread_seq_write(void);
__device__ void prefetch_seq(int64_t start_lba,int64_t embed_num,uintptr_t *dev_addr);
__device__ void writeback_seq(int64_t start_lba,int64_t embed_num,uintptr_t *dev_addr);



#endif