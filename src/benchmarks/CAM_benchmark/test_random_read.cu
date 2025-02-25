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
#include "spdk_interface.h"
#include <QDMAController.hpp>


const int64_t embed_num = 100000;
//uintptr_t dev_addr[embed_num];
//static GPUMemCtl* gpuMemCtl;
static const int64_t lba_size = 512;
static int64_t embed_entry_width ;
static int64_t embed_entry_lba;



__inline__ uint64_t get_tsc()
{
    uint64_t a, d;
    __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
    return (d << 32) | a;
}
 
__inline__ uint64_t get_tscp(void)
{
  uint32_t lo, hi;
  // take time stamp counter, rdtscp does serialize by itself, and is much cheaper than using CPUID
  __asm__ __volatile__ (
      "rdtscp" : "=a"(lo), "=d"(hi)
      );
  return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

__inline__ uint64_t cycles_2_ns(uint64_t cycles, uint64_t hz)
{
  return cycles * (1000000000.0 / hz);
}

uint64_t get_cpu_freq()
{
    FILE *fp=popen("lscpu | grep CPU | grep MHz | awk  {'print $3'}","r");
    if(fp == nullptr)
        return 0;
 
    char cpu_mhz_str[200] = { 0 };
    fgets(cpu_mhz_str,80,fp);
    fclose(fp);
 
    return atof(cpu_mhz_str) * 1000 * 1000;

}




static void run_task_function_test() {
    u_int64_t* embed_id = (u_int64_t*)malloc(embed_num*sizeof(u_int64_t));
    //launch_idle_kernel();
    void* gem_memory = alloc_gpu(embed_num*4096);
    for (int64_t i = 0; i < embed_num; i++) {
        embed_id[i] = i;
        //dev_addr[i] = (uintptr_t)gem_memory + i * embed_entry_width;
    }
    // int buffer[1024];
    // int buffer_fake[1024];
    // int buffer2[1024];
    // for(int i=0;i<1024;i++){
    //     buffer[i]=i;
    //     buffer_fake[i] =0;
    // }
    std::cout<<" begin!"<<std::endl;
    std::random_shuffle(embed_id, embed_id + embed_num-1);
    //std::random_shuffle(dev_addr, dev_addr + embed_num);

    // cudaMemcpy((void*)(dev_addr[2]), buffer, 1024 * sizeof(int), cudaMemcpyHostToDevice);
    // task_submit_write(embed_num, (u_int64_t)embed_id, dev_addr);
    // clear_wait_flag_write();
    // cudaMemcpy((void*)(dev_addr[2]), buffer_fake, 1024 * sizeof(int), cudaMemcpyHostToDevice);
    // task_submit(embed_num, (u_int64_t)embed_id, dev_addr);
    // clear_wait_flag();
    // cudaMemcpy(buffer2, (void*)(dev_addr[2]), 1024 * sizeof(int) , cudaMemcpyDeviceToHost);
    clock_t start,finish;
    std::cout<<"loop"<<std::endl;
    double sum=0;
        uint64_t beg_tsc, end_tsc, middle_tsc;
        beg_tsc = get_tscp();
    for(int i=0;i<10;i++){

            //seq_read_submit(0, embed_num,(uintptr_t)gem_memory);
            //cam_gemm_read((u_int64_t *)embed_id, embed_num,(uintptr_t)gem_memory);

            //clear_wait_flag();
            //seq_write_submit(0, embed_num,(uintptr_t)gem_memory);
            cam_gemm_read((u_int64_t *)embed_id, embed_num,(uintptr_t)gem_memory);
        
            clear_wait_flag();

    }
    end_tsc = get_tscp();
        sum = 1.0*(end_tsc-beg_tsc)/ 2.2;
        printf("time cost : %lf ms\n",1.0*sum/1000000);
        std::cout<<"bandwidth: "<< embed_num*4096*10 / sum  << "GB/s" <<std::endl;
    
    // g_namespaces.resize(1);

    //printf("Start to submit task\n");
    free_gpu(gem_memory);
    // auto time_start = std::chrono::high_resolution_clock::now();

    
    
    // auto time_end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start);
    // printf("Time: %f\n", time_span.count());
    // printf("bandwdth : %lf GB/s\n",embed_num*4/time_span.count()/1024/1024);
}

int main(int argc, char** argv) {
    
    cam_init(4096);
    run_task_function_test();    
    cam_clean_up();

    return 0;
}

/*
nvcc -o test_random_read  -I /home/szy/yzh_hyprion/spdk_interface  -L /home/szy/yzh_hyprion/spdk_interface -lgpussd_baseline test_random_read.cu
*/
