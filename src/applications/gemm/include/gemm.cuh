#pragma once
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include "lightbam.cuh"
uint64_t parse_offset(char *str)
{
    int len = strlen(str);
    char unit = ' ';
    if (!isdigit(str[len - 1]))
    {
        unit = str[len - 1];
        str[len - 1] = '\0';
    }
    uint64_t offset = atoll(str);
    if (unit == 'K' || unit == 'k')
    {
        offset = offset * 1024;
    }
    if (unit == 'M' || unit == 'm')
    {
        offset = offset * 1024 * 1024;
    }
    if (unit == 'G' || unit == 'g')
    {
        offset = offset * 1024 * 1024 * 1024;
    }
    if (unit == 'T' || unit == 't')
    {
        offset = offset * 1024 * 1024 * 1024 * 1024;
    }
    return offset;
}

class PinnedBuffer
{
private:
    void *iobuf;
    aeolus_dev_mem_context *iobuf_ctx;
    uint64_t *h_iobuf_phys;
    uint64_t *d_iobuf_phys;
    uint64_t *prp_list;
    uint64_t *h_prp_phys;
    uint64_t *d_prp_phys;
    uint64_t max_io_size;
    Device *dev;

public:
    PinnedBuffer(Device* dev, uint64_t size, uint64_t max_io_size = 0)
    {
        int ret = dev->alloc_device_memory(
            &iobuf, &iobuf_ctx, size, &h_iobuf_phys
        );
        if (ret != 0)
        {
            AEOLUS_LOG_ERROR("Failed to allocate device memory for IO buffer: %s", strerror(ret));
            exit(-1);
        }
        size_t iobuf_phys_size = size / AEOLUS_DEVICE_PGSIZE * sizeof(uint64_t);
        AEOLUS_CUDA_CHECK(cudaMalloc(&d_iobuf_phys, iobuf_phys_size));
        AEOLUS_CUDA_CHECK(cudaMemcpy(d_iobuf_phys, h_iobuf_phys, iobuf_phys_size, cudaMemcpyHostToDevice));

        if (max_io_size > AEOLUS_HOST_PGSIZE * 2)
        {
            uint64_t prp_list_size = size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t);
            AEOLUS_LOG_INFO("Allocating PRP buffer.");
            dev->alloc_host_memory((void **)&prp_list, prp_list_size, &h_prp_phys);

            // Fill in PRP table.
            for (int i = 0; i < size / AEOLUS_DEVICE_PGSIZE; i++)
            {
                for (int j = 0; j < AEOLUS_DEVICE_PGSIZE / AEOLUS_HOST_PGSIZE; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }
                    prp_list[i * AEOLUS_DEVICE_PGSIZE / AEOLUS_HOST_PGSIZE + j - 1] = 
                        h_iobuf_phys[i] + j * AEOLUS_HOST_PGSIZE;
                }
            }

            // Move PRP physical address to GPU.
            size_t prp_phys_size = CEIL(prp_list_size, AEOLUS_HOST_PGSIZE) * sizeof(uint64_t);
            AEOLUS_CUDA_CHECK(cudaMalloc((void **)&d_prp_phys, prp_phys_size));
            AEOLUS_CUDA_CHECK(cudaMemcpy(d_prp_phys, h_prp_phys, prp_phys_size, cudaMemcpyHostToDevice));
        }
        this->max_io_size = max_io_size;
        this->dev = dev;
    }

    Request create_request(uint64_t offset, uint64_t start_lb = 0, int num_items = 0)
    {
        if (num_items == 0)
            num_items = max_io_size / AEOLUS_LB_SIZE;
        Request req(start_lb, num_items);
        req.dest_addr = h_iobuf_phys[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
        req.next_addr = offset / max_io_size;
        if (max_io_size <= AEOLUS_HOST_PGSIZE * 2)
        {
            offset += AEOLUS_HOST_PGSIZE;
            req.next_addr = h_iobuf_phys[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
        }
        return req;
    }

    uint64_t *get_iobuf_phys()
    {
        return h_iobuf_phys;
    }

    uint64_t *get_d_iobuf_phys()
    {
        return d_iobuf_phys;
    }

    uint64_t *get_d_prp_phys()
    {
        return d_prp_phys;
    }

    operator void *()
    {
        return iobuf;
    }

    ~PinnedBuffer()
    {
        if (max_io_size > AEOLUS_HOST_PGSIZE * 2)
        {
            dev->free_host_memory(prp_list, h_prp_phys);
            AEOLUS_CUDA_CHECK(cudaFree(d_prp_phys));
        }
        dev->free_device_memory(iobuf_ctx);
    }
};