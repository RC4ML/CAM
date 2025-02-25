#include "lightbam.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cublas_v2.h>
#include <algorithm>
#include "gemm.cuh"
#include <vector>
#include "spdk_read.h"

typedef float fp_t;
int main(int argc, char *argv[])
{
    if (argc != 10)
    {
        printf("Usage: %s m n k a_offset b_offset c_offset block_size max_io_size num_ssds\n", argv[0]);
        return 1;
    }
    int m = parse_offset(argv[1]);
    int n = parse_offset(argv[2]);
    int k = parse_offset(argv[3]);
    uint64_t a_offset = parse_offset(argv[4]);
    uint64_t b_offset = parse_offset(argv[5]);
    uint64_t c_offset = parse_offset(argv[6]);
    uint64_t block_size = parse_offset(argv[7]);
    uint64_t max_io_size = parse_offset(argv[8]);
    int num_ssds = atoi(argv[9]);
    if (m % block_size != 0 || n % block_size != 0)
    {
        std::cout<<"m and n must be a multiple of block_size"<<std::endl;
        return 1;
    }
    int m_blocks = m / block_size;
    int n_blocks = n / block_size;
    if (block_size * sizeof(fp_t) % max_io_size != 0)
    {
        std::cout<<"block_size * sizeof(fp_t) must be a multiple of max_io_size"<<std::endl;
        return 1;
    }
    int num_queues_per_ssd = CEIL(block_size * k * sizeof(fp_t), num_ssds * 4096 * max_io_size) + 1;
    fp_t *a0, *a1, *b0, *b1, *c0, *c1;
    printf("max_io_size = %ld\n", max_io_size); 
    cam_init(max_io_size);
    // a0 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    // a1 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    // b0 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    // b1 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    // c0 = (fp_t*)alloc_gpu(block_size * block_size * sizeof(fp_t));
    // c1 = (fp_t*)alloc_gpu(block_size * block_size * sizeof(fp_t));
    a1 = (fp_t*)alloc_pinmemory(block_size * k * sizeof(fp_t));
    b1 = (fp_t*)alloc_pinmemory(block_size * k * sizeof(fp_t));
    c1 = (fp_t*)alloc_pinmemory(block_size * block_size * sizeof(fp_t));
    if(a1 == NULL || b1 == NULL || c1 == NULL){
        printf("alloc pin memory  failed\n");
        
    }  
    cudaHostRegister(a1, block_size * k * sizeof(fp_t), cudaHostRegisterDefault);
    cudaHostRegister(b1, block_size * k * sizeof(fp_t), cudaHostRegisterDefault);
    cudaHostRegister(c1, block_size * block_size * sizeof(fp_t), cudaHostRegisterDefault);
    cudaMalloc(&a0, block_size * k * sizeof(fp_t));
    cudaMalloc(&b0, block_size * k * sizeof(fp_t));
    cudaMalloc(&c0, block_size * block_size * sizeof(fp_t));
    int num_reqs = CEIL(block_size * k * sizeof(fp_t), max_io_size);
    u_int64_t *h_reqs = (u_int64_t *)malloc(num_reqs * sizeof(u_int64_t));
    u_int64_t *h_reqs2 = (u_int64_t *)malloc(num_reqs * sizeof(u_int64_t));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    fp_t alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop, gemm_start, gemm_stop;
    cudaStream_t streama,streamb,streamc;
    cudaStreamCreate(&streama);
    cudaStreamCreate(&streamb);
    cudaStreamCreate(&streamc);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gemm_start);
    cudaEventCreate(&gemm_stop);
    cudaEventRecord(start, 0);
    float gemm_ms = 0;
    for (int j = 0; j < n_blocks; j++)
    {
        for (int i = 0; i < num_reqs; i++)
        {
            uint64_t offset = 1ll * i * max_io_size / sizeof(fp_t);
            int row = offset / block_size;
            int col = j * block_size + offset % block_size;
            h_reqs[i] = (b_offset + (1ll * row * n + col) * sizeof(fp_t)) / AEOLUS_LB_SIZE;
        }
        cam_gemm_read(h_reqs,num_reqs,(uintptr_t)b1);
        clear_wait_flag();
        //std::swap(b0, b1);
        cudaMemcpyAsync(b0, b1, block_size * k * sizeof(fp_t), cudaMemcpyHostToDevice, streamb);
        for (int i = -1; i <= m_blocks+2; i++)
        {
            if (i >= 0 && i < m_blocks)
            {
                // clear_wait_flag();
                cudaMemcpyAsync(a0, a1, block_size * k * sizeof(fp_t), cudaMemcpyHostToDevice, streama);
                //std::swap(a0, a1);
            }
            if (i  < m_blocks -1)    //read phase
            {
                for (int l = 0; l < num_reqs; l++)
                {
                    uint64_t offset = 1ll * l * max_io_size / sizeof(fp_t);
                    h_reqs[l] = (a_offset + ((i + 1) * block_size * k + offset) * sizeof(fp_t)) / AEOLUS_LB_SIZE;
                }
                cam_gemm_read(h_reqs,num_reqs,(uintptr_t)a1);
                clear_wait_flag();
            }
            if(i>=2 && i<= m_blocks+1){
                // std::swap(c0, c1);
                cudaMemcpyAsync(c0, c1, block_size * block_size * sizeof(fp_t), cudaMemcpyHostToDevice, streamc);
            }
            if (i >= 3)    //write phase
            {
                // if (i  >= 4)
                // {
                //     clear_wait_flag_write();
                // }
                
                int num_reqs = CEIL(block_size * block_size * sizeof(fp_t), max_io_size);
                for (int l = 0; l < num_reqs; l++)
                {
                    uint64_t offset = 1ll * l * max_io_size / sizeof(fp_t);
                    int row = (i - 1) * block_size + offset / block_size;
                    int col = j * block_size + offset % block_size;
                    h_reqs2[l] = (c_offset + (1ll * row * n + col) * sizeof(fp_t)) / AEOLUS_LB_SIZE;
                    // h_reqs2[l] = l;
                }
                cudaStreamSynchronize(streamc);
                cam_gemm_write(h_reqs2,num_reqs,(uintptr_t)c1);
                clear_wait_flag_write();
                
            }
            
            if (i >= 1 && i <= m_blocks)   //gemm compute phase
            {
                cudaEventRecord(gemm_start, 0);
                cudaStreamSynchronize(streama);
                cudaStreamSynchronize(streamb);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, block_size, block_size, k, &alpha, b0, CUDA_R_32F, block_size, a0, CUDA_R_32F, k, &beta, c0, CUDA_R_32F, block_size, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                cudaEventRecord(gemm_stop, 0);
                cudaEventSynchronize(gemm_stop);
                float ms;
                cudaEventElapsedTime(&ms, gemm_start, gemm_stop);
                gemm_ms += ms;
            }
         }
        // clear_wait_flag_write();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("m = %d, n = %d, k = %d, block_size = %ld, time = %f ms, tflops = %f\n", m, n, k, block_size, ms, 2.0 * m * n * k / ms / 1e9);
    printf("gemm time = %f ms, num_ssds = %d, max_io_size = %ld, num_queues = %d\n", gemm_ms, num_ssds, max_io_size, num_queues_per_ssd);
    printf("%d %ld %d %ld %f %f %d\n", n, block_size, num_ssds, max_io_size, gemm_ms, ms, num_queues_per_ssd);
    cublasDestroy(handle);
    free_pinmemory(a1);
    free_pinmemory(b1);
    free_pinmemory(c1);
    cudaFree(a0);
    cudaFree(b0);
    cudaFree(c0);
    cudaStreamDestroy(streama);
    cudaStreamDestroy(streamb);
    cudaStreamDestroy(streamc);
    free(h_reqs);
    cam_clean_up();
    return 0;
}