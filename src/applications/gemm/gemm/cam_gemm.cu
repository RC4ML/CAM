#include "lightbam.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cublas_v2.h>
#include <algorithm>
#include "gemm.cuh"
#include <vector>
#include "gpussd_baseline.h"

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
    cam_init(max_io_size);
    a0 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    a1 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    b0 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    b1 = (fp_t*)alloc_gpu(block_size * k * sizeof(fp_t));
    c0 = (fp_t*)alloc_gpu(block_size * block_size * sizeof(fp_t));
    c1 = (fp_t*)alloc_gpu(block_size * block_size * sizeof(fp_t));
    if(a0 == NULL || a1 == NULL || b0 == NULL || b1 == NULL || c0 == NULL || c1 == NULL){
        printf("alloc gpu memory failed\n");
        return 1;
    }
    int num_reqs = CEIL(block_size * k * sizeof(fp_t), max_io_size);
    u_int64_t *h_reqs = (u_int64_t *)malloc(num_reqs * sizeof(u_int64_t));
    u_int64_t *h_reqs2 = (u_int64_t *)malloc(num_reqs * sizeof(u_int64_t));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    fp_t alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop, gemm_start, gemm_stop;
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
        for (int i = -1; i <= m_blocks; i++)
        {
            if (i >= 0 && i < m_blocks)
            {
                //clear_wait_flag();
                std::swap(a0, a1);
            }
            if (i + 1 < m_blocks)
            {
                for (int l = 0; l < num_reqs; l++)
                {
                    uint64_t offset = 1ll * l * max_io_size / sizeof(fp_t);
                    h_reqs[l] = (a_offset + ((i + 1) * block_size * k + offset) * sizeof(fp_t)) / AEOLUS_LB_SIZE;
                }
                cam_gemm_read(h_reqs,num_reqs,(uintptr_t)a1);
                clear_wait_flag();
            }
            if (i - 1 >= 0)
            {
                // if (i - 2 >= 0)
                // {
                //     clear_wait_flag_write();
                // }
                std::swap(c0, c1);
                int num_reqs = CEIL(block_size * block_size * sizeof(fp_t), max_io_size);
                for (int l = 0; l < num_reqs; l++)
                {
                    uint64_t offset = 1ll * l * max_io_size / sizeof(fp_t);
                    int row = (i - 1) * block_size + offset / block_size;
                    int col = j * block_size + offset % block_size;
                    h_reqs2[l] = (c_offset + (1ll * row * n + col) * sizeof(fp_t)) / AEOLUS_LB_SIZE;
                }
                cam_gemm_write(h_reqs2,num_reqs,(uintptr_t)c1);
                clear_wait_flag_write();
            }
            if (i == 0)
            {
                std::swap(b0, b1);
            }
            if (i >= 0 && i < m_blocks)
            {
                cudaEventRecord(gemm_start, 0);
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
    free_gpu(a0);
    free_gpu(a1);
    free_gpu(b0);
    free_gpu(b1);
    free_gpu(c0);
    free_gpu(c1);
    free(h_reqs);
    cam_clean_up();
    return 0;
}