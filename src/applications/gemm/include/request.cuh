#ifndef __AEOLUS_REQUEST_CUH
#define __AEOLUS_REQUEST_CUH

#include <stdint.h>
#include "util.cuh"

enum aeolus_access_dir
{
    AEOLUS_DIR_READ   = 0,
    AEOLUS_DIR_WRITE  = 1
};

class Request
{
public:
    uint64_t    start_lb;  // starting logical block
    uint64_t    dest_addr; // physical address (pinned buffer) / virtual address (non-pinned buffer) of destination
    uint64_t    next_addr; // only valid for pinned buffer, next page of dest_addr (io_size <= 8KB) / prp list offset (io_size > 8KB)
    int         num_items; // number of logical blocks

    inline __host__ __device__ Request(uint64_t start_lb, int num_items)
    {
        this->start_lb = start_lb;
        this->num_items = num_items;
        // You may need to call cudaLimitMallocHeapSize beforehand in this implementation.
        // this->dest_addr = (uint64_t*)malloc(sizeof(uint64_t)*num_items);
    }

    inline __host__ __device__ bool operator<(const Request& other) const
    {
        return this->start_lb < other.start_lb;
    }

    inline __host__ __device__ ~Request()
    {
        // free(this->dest_addr);
    }
};

#endif