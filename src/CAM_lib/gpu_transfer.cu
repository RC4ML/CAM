#include "gpu_transfer.cuh"

//static  struct SmDevice2DeviceSemaphoreDeviceHandle d_sm;
u_int64_t read_block_num = 1000000UL;
//* read arguments
static Host2DeviceSemaphore h_sm;
static u_int64_t *h_data;
__device__ static u_int64_t *d_data;
std::vector<u_int64_t> a;

static u_int64_t *h_submit_info;
__device__ static u_int64_t *d_submit_info;


//* D2H Semaphore arguments
__device__ uint64_t* D2H_inboundSemaphoreId;
__device__ uint64_t* D2H_expectedInboundSemaphore;
__device__ uint64_t* D2H_outboundSemaphoreId;
__device__ uint64_t* D2H_outboundSemaphoreValue;
__device__ uint64_t* D2H_total_num;

//* write arguments
static Host2DeviceSemaphore h_sm_2;
static u_int64_t *h_data_2;
__device__ static u_int64_t *d_data_2;
std::vector<u_int64_t> a_2;

static u_int64_t *h_submit_info_2;
__device__ static u_int64_t *d_submit_info_2;


//* D2H Semaphore arguments
__device__ uint64_t* D2H_inboundSemaphoreId_2;
__device__ uint64_t* D2H_expectedInboundSemaphore_2;
__device__ uint64_t* D2H_outboundSemaphoreId_2;
__device__ uint64_t* D2H_outboundSemaphoreValue_2;
__device__ uint64_t* D2H_total_num_2;

void SemaphoreInit(cudaStream_t stream1)
{
    void* tmp;
    cudaMalloc(&(tmp),sizeof(u_int64_t));
    cudaMemcpyToSymbol(D2H_outboundSemaphoreValue, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    cudaMalloc(&(tmp),sizeof(u_int64_t));
    cudaMemcpyToSymbol(D2H_expectedInboundSemaphore, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    void *tmp_h_data;
    cudaHostAlloc((void**)&tmp_h_data, sizeof(u_int64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&(tmp), (u_int64_t *)tmp_h_data, 0);
    cudaMemcpyToSymbol(D2H_outboundSemaphoreId, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    
    void *tmp_num;
    cudaHostAlloc((void**)&tmp_num, sizeof(u_int64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&(D2H_total_num), tmp_num, 0);


    h_sm.ConnectToStream(stream1);
    h_sm.ConnectToDeviceSemaphore(tmp_h_data,(u_int64_t *)tmp_num);
    tmp = (uint64_t*)h_sm.GetoutboundSemaphore();
    cudaMemcpyToSymbol(D2H_inboundSemaphoreId, &(tmp), sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    
    cudaMalloc(&(tmp),sizeof(u_int64_t));
    cudaMemcpyToSymbol(D2H_outboundSemaphoreValue_2, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    cudaMalloc(&(tmp),sizeof(u_int64_t));
    cudaMemcpyToSymbol(D2H_expectedInboundSemaphore_2, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    void *tmp_h_data_2;
    cudaHostAlloc((void**)&tmp_h_data_2, sizeof(u_int64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&(tmp), (u_int64_t *)tmp_h_data_2, 0);
    cudaMemcpyToSymbol(D2H_outboundSemaphoreId_2, &tmp, sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    
    
    cudaHostAlloc((void**)&tmp_num, sizeof(u_int64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&(D2H_total_num_2), tmp_num, 0);


    h_sm_2.ConnectToStream(stream1);
    h_sm_2.ConnectToDeviceSemaphore(tmp_h_data_2,(u_int64_t *)tmp_num);
    tmp = (uint64_t*)h_sm_2.GetoutboundSemaphore();
    cudaMemcpyToSymbol(D2H_inboundSemaphoreId_2, &(tmp), sizeof(uint64_t),0, cudaMemcpyHostToDevice);
}

__device__ void prefetch(int64_t embed_num,uintptr_t *dev_addr)
{
    __syncthreads();
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        
        d_submit_info[0] = 1;
        d_submit_info[1] = embed_num;
        d_submit_info[2] = (uint64_t)dev_addr;

        *D2H_outboundSemaphoreValue += 1;
        //printf("D2H_outboundSemaphoreValue: %ld\n",*D2H_outboundSemaphoreValue);
        *D2H_outboundSemaphoreId = *D2H_outboundSemaphoreValue;
        //printf("D2H_outboundSemaphoreId: %ld\n",*D2H_outboundSemaphoreId);
    }
}

__device__ void prefetch_syncronize(void)
{
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        //printf("leadind thread wait\n");
        (*D2H_expectedInboundSemaphore) += 1;
        uint64_t value;
        uint64_t value2= (*D2H_expectedInboundSemaphore);
        //printf("value2: %ld\n",value2);
        while(true){
            value= atomicMin((unsigned long long int*)D2H_inboundSemaphoreId,(unsigned long long int)value2);
            if(value >= value2)
                break;  
            uint64_t start = 0;
            while (start++ < 100000);
        }

        //printf("leadind thread wait done\n");
    }
    __syncthreads();
}


void polling_thread(void)
{
    while(1){
        h_sm.wait();
        uint64_t embed_num = h_submit_info[1];
        uintptr_t *gem_memory = (uintptr_t *)(h_submit_info[2]);
        cam_gemm_read(h_data, embed_num,(uintptr_t)gem_memory);
        clear_wait_flag();
        h_sm.signal();
    }
}


void polling_thread_seq(void)
{
    while(1){
        h_sm.wait();
        u_int64_t start_lba = h_submit_info[0];
        uint64_t embed_num = h_submit_info[1];
        uintptr_t *gem_memory = (uintptr_t *)(h_submit_info[2]);
        seq_read_submit(start_lba,embed_num,(uintptr_t)gem_memory);
        clear_wait_flag();
        h_sm.signal();
    }
}

void polling_thread_seq_write(void)
{
    while(1){
        h_sm.wait();
        u_int64_t start_lba = h_submit_info[0];
        uint64_t embed_num = h_submit_info[1];
        uintptr_t *gem_memory = (uintptr_t *)(h_submit_info[2]);
        seq_write_submit(start_lba,embed_num,(uintptr_t)gem_memory);
        clear_wait_flag_write();
        h_sm.signal();
    }
}

__device__ void prefetch_seq(int64_t start_lba,int64_t embed_num,uintptr_t *dev_addr)
{
    __syncthreads();
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        
        d_submit_info[0] = start_lba;
        d_submit_info[1] = embed_num;
        d_submit_info[2] = (uint64_t)dev_addr;

        *D2H_outboundSemaphoreValue += 1;
        //printf("D2H_outboundSemaphoreValue: %ld\n",*D2H_outboundSemaphoreValue);
        *D2H_outboundSemaphoreId = *D2H_outboundSemaphoreValue;
        //printf("D2H_outboundSemaphoreId: %ld\n",*D2H_outboundSemaphoreId);
    }
}

__device__ void writeback_seq(int64_t start_lba,int64_t embed_num,uintptr_t *dev_addr)
{
    __syncthreads();
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        
        d_submit_info_2[0] = start_lba;
        d_submit_info_2[1] = embed_num;
        d_submit_info_2[2] = (uint64_t)dev_addr;

        *D2H_outboundSemaphoreValue_2 += 1;
        //printf("D2H_outboundSemaphoreValue: %ld\n",*D2H_outboundSemaphoreValue);
        *D2H_outboundSemaphoreId_2 = *D2H_outboundSemaphoreValue_2;
        //printf("D2H_outboundSemaphoreId: %ld\n",*D2H_outboundSemaphoreId);
    }
}


void Init(u_int32_t access_size,cudaStream_t stream1)
{
    SemaphoreInit(stream1);
    // 在主机端分配页锁内存（零拷贝内存）
    cudaHostAlloc((void**)&h_data, read_block_num * sizeof(u_int64_t), cudaHostAllocMapped);
    // 获取对应的设备指针
    cudaHostGetDevicePointer((void**)&d_data, h_data, 0);

    cudaHostAlloc((void**)&h_submit_info, 3 * sizeof(u_int64_t), cudaHostAllocMapped);
    
    void* tmp;
    cudaHostGetDevicePointer((void**)&tmp, h_submit_info, 0);
    cudaMemcpyToSymbol(d_submit_info, &(tmp), sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    
    cudaHostAlloc((void**)&h_data_2, read_block_num * sizeof(u_int64_t), cudaHostAllocMapped);
    // 获取对应的设备指针
    cudaHostGetDevicePointer((void**)&d_data_2, h_data_2, 0);

    cudaHostAlloc((void**)&h_submit_info_2, 3 * sizeof(u_int64_t), cudaHostAllocMapped);
    

    cudaHostGetDevicePointer((void**)&tmp, h_submit_info_2, 0);
    cudaMemcpyToSymbol(d_submit_info_2, &(tmp), sizeof(uint64_t),0, cudaMemcpyHostToDevice);
    //cam_init(access_size);
    init_myKernel<<<1, 1,0,stream1>>>();
    
    std::cout<<"init done"<<std::endl;
}

extern "C" __global__ void init_myKernel(void) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 进行一些计算并更新设备内存数据
    if(idx == 0)
    {
        printf("init kernel\n");
        *(D2H_expectedInboundSemaphore) =0;
        *(D2H_outboundSemaphoreValue)=0;

        *(D2H_expectedInboundSemaphore_2) =0;
        *(D2H_outboundSemaphoreValue_2)=0;
    }
    
    /*
    /* computation 
    */

    __syncthreads();
}

uint64_t* get_d_data(void){return d_data;}

uint64_t* get_d_data_write(void){return d_data_2;}


void polling_thread_write(void)
{
    while(1){
        h_sm_2.wait();
        uint64_t embed_num = h_submit_info_2[1];
        // uint64_t* embed_id = h_data;//(uint64_t*)(h_submit_info[1]);
        uintptr_t *gem_memory = (uintptr_t *)(h_submit_info_2[2]);
        // printf("embed_num: %ld\n",h_submit_info[0]);
        // printf("embed_id: %ld\n",h_submit_info[1]);
        // printf("gem_memory: %ld\n",h_submit_info[2]);
        //cout<< "total number: "<< h_sm.GetTotalNumber()<<endl;
        //void* gem_memory = alloc_gpu(read_block_num*4096);
        //cam_gemm_read(h_data,read_block_num,(uintptr_t)gem_memory);
        //clear_wait_flag();
        cam_gemm_write(h_data_2, embed_num,(uintptr_t)gem_memory);
        //std::cout<<"submit done"<<std::endl;
        clear_wait_flag_write();
        h_sm_2.signal();
    }
}

__device__ void writeback(int64_t embed_num,uintptr_t *dev_addr)
{
    __syncthreads();
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        
        d_submit_info_2[0] = 1;
        d_submit_info_2[1] = embed_num;
        d_submit_info_2[2] = (uint64_t)dev_addr;

        *D2H_outboundSemaphoreValue_2 += 1;
        //printf("D2H_outboundSemaphoreValue: %ld\n",*D2H_outboundSemaphoreValue);
        *D2H_outboundSemaphoreId_2 = *D2H_outboundSemaphoreValue_2;
        //printf("D2H_outboundSemaphoreId: %ld\n",*D2H_outboundSemaphoreId);
    }
}

__device__ void writeback_syncronize(void)
{
    if((threadIdx.x + blockIdx.x * blockDim.x) == 0)
    {
        //printf("leadind thread wait\n");
        (*D2H_expectedInboundSemaphore_2) += 1;
        uint64_t value;
        uint64_t value2= (*D2H_expectedInboundSemaphore_2);
        //printf("value2: %ld\n",value2);
        while(true){
            value= atomicMin((unsigned long long int*)D2H_inboundSemaphoreId_2,(unsigned long long int)value2);
            if(value >= value2)
                break;   
            // uint64_t start = 0;
            // while (start++ < 10000000000);
        }

        //printf("leadind thread wait done\n");
    }
    __syncthreads();
}