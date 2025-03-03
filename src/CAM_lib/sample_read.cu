#include <iostream>
#include <math.h>
#include <unistd.h>
#include <string>
#include <fcntl.h>
#include <random>
#include <chrono>
#include <thread>
#include <cstdint>
#include<ctime>
#include "gpu_transfer.cuh"
#include "CAM_interface.h"

__inline__ uint64_t get_tscp(void)
{
  uint32_t lo, hi;
  // take time stamp counter, rdtscp does serialize by itself, and is much cheaper than using CPUID
  __asm__ __volatile__ (
      "rdtscp" : "=a"(lo), "=d"(hi)
      );
  return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}



__global__ void myKernel( u_int64_t* d_data_dev,u_int64_t* d_data,uint64_t* dev_addr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=0;(256*i+idx)<1000000UL;i++)
        d_data[256*i+idx]=d_data_dev[256*i+idx];
    for(int i=0;i<10;i++){
        prefetch(1000000UL,dev_addr);
        prefetch_syncronize();
    }
}


int main(int argc, char** argv) {
    
    cam_init(4096);
    u_int64_t* embed_id = (u_int64_t*)malloc(10000000UL*sizeof(u_int64_t));
    //launch_idle_kernel();
    for (int64_t i = 0; i < 10000000UL; i++) {
        embed_id[i] = i;
    }
    std::random_shuffle(embed_id, embed_id + 10000000UL-1);
    u_int64_t* embed_id_dev;
    cudaMalloc(&embed_id_dev,10000000UL*sizeof(u_int64_t));
    cudaMemcpy(embed_id_dev,embed_id,10000000UL*sizeof(u_int64_t),cudaMemcpyHostToDevice);
    cudaStream_t stream1,stream2;
    cudaError_t result;
    result = cudaStreamCreate(&stream1);
    result = cudaStreamCreate(&stream2);
    Init(4096,stream1);
    void* gem_memory = alloc_gpu(1000000UL*4096);
    u_int64_t* p_d = get_d_data();
    std::thread th(polling_thread);
    double sum=0;
    uint64_t beg_tsc, end_tsc, middle_tsc;
    beg_tsc = get_tscp();
    myKernel<<<1, 256,0,stream2>>>(embed_id_dev,p_d,(uint64_t*)gem_memory);
    cudaDeviceSynchronize();
    end_tsc = get_tscp();
    sum = 1.0*(end_tsc-beg_tsc)/ 2.2;
    printf("time cost : %lf ms\n",1.0*sum/1000000);
    std::cout<<"bandwidth: "<< 1000000UL*4096*10 / sum  << "GB/s" <<std::endl;
    
    cam_clean_up();
    printf("done\n");
    return 0;
}

/*

nvcc  -dc  -o test_read.o   -I /home/szy/application/cam  -L /home/szy/application/cam -lspdk_interface -lgpu_transfer test.cu
nvcc -dlink test_read.o b.o -o link.o
nvcc -o test_read_executable   -L/home/szy/application/cam -lspdk_interface -lgpu_transfer test_read.o
nvcc -o test_read_executable test_read.o -L/home/szy/application/cam -lgpu_transfer -lspdk_interface
nvcc -rdc=true test.cu gpu_transfer.o -o main -I /home/szy/application/cam  -L /home/szy/application/cam -lspdk_interface
*/
