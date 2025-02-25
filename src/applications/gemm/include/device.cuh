#ifndef __AEOLUS_DEVICE_CUH
#define __AEOLUS_DEVICE_CUH

#define _CUDA

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "log.cuh"
#include "util.cuh"
#include "ioctl.h"
#include "queue.cuh"

struct aeolus_dev_mem_context
{
    void *ptr;
    void *ioaddrs;
};

/**
 * @brief Abstraction of an SSD device.
 * 
 */
class Device
{
private:
    void        *admin_queue_ptr;
    uint64_t    *admin_queue_phys_addr;

public:
    int         ssd_id;
    int         device_fd;
    void        *reg_ptr;
    uint32_t    max_queue_num;
    uint64_t    max_lb_num;
    uint32_t    max_io_size;

    AdminQueuePair  *admin_qp;
    uint32_t        active_ns;
    
    std::vector<int32_t> free_qps;

    /**
     * @brief Construct a new Device object.
     * 
     * @param ssd_id ID of the SSD, typically the number in /dev/libnvm*
     */
    Device(int ssd_id);

    ~Device();

    /**
     * @brief Allocate a pinned host memory buffer with physical address provided.
     * 
     * @param ptr Buffer pointer to be allocated. 
     * @param size Size of the buffer.
     * @param phys_addr A physical address list returned. Each entry of the list is a 4KB page.
     * @return Allocation result. Can be read by strerror. 
     */
    int alloc_host_memory(void **ptr, uint64_t size, uint64_t** phys_addr);

    /**
     * @brief Free a pinned host memory buffer.
     * 
     * @param ptr Buffer pointer.
     * @param phys_addr Physical address list pointer. 
     */
    void free_host_memory(void *ptr, uint64_t* phys_addr);

    /**
     * @brief Allocate a pinned device memory buffer with physical address provided. The buffer will be 64KB-aligned.
     * 
     * @param ptr Buffer pointer to be allocated. 
     * @param context A context pointer which could be used for freeing the buffer.
     * @param size Size of the buffer.
     * @param phys_addr A physical address list returned. Each entry of the list is a 64KB page.
     * @return Allocation result. Can be read by strerror.  
     */
    int alloc_device_memory(void **ptr, aeolus_dev_mem_context** context, uint64_t size, uint64_t** phys_addr);

    /**
     * @brief Free a pinned device memory buffer.
     * 
     * @param context Device buffer context.
     */
    void free_device_memory(aeolus_dev_mem_context* context);
};

#endif