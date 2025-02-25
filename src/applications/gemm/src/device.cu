#include "device.cuh"

Device::Device(int ssd_id)
{

    // Open file and map BAR0 of SSD

    this->ssd_id = ssd_id;
    AEOLUS_LOG_INFO("Setting up device %d", ssd_id);
    char device_path[64];
    sprintf(device_path, "/dev/libnvm%d", ssd_id);
    device_fd = open(device_path, O_RDWR);
    if (device_fd < 0)
    {
        AEOLUS_LOG_ERROR("Failed to open: %s", strerror(errno));
        exit(1);
    }
    reg_ptr = mmap(NULL, NVME_BAR0_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, device_fd, 0);
    if (reg_ptr == MAP_FAILED)
    {
        AEOLUS_LOG_ERROR("Failed to mmap: %s\n", strerror(errno));
        exit(1);
    }
    AEOLUS_CUDA_CHECK(cudaHostRegister(reg_ptr, NVME_BAR0_SIZE, cudaHostRegisterIoMemory));

    // Reset controller.

    uint64_t reg_ptr_uint = (uint64_t)reg_ptr;
    *(uint32_t *)(reg_ptr_uint + NVME_REG_CC) &= ~NVME_REG_CC_EN;
    while (*(uint32_t volatile *)(reg_ptr_uint + NVME_REG_CSTS) & NVME_REG_CSTS_RDY)
        ;
    AEOLUS_LOG_INFO("Reset done.");

    // Set admin queue attributes.

    int ret = alloc_host_memory(&admin_queue_ptr, 2*AEOLUS_HOST_PGSIZE, &admin_queue_phys_addr);
    if (ret != 0)
    {
        AEOLUS_LOG_ERROR("Allocate admin queue memory failed: %s", strerror(ret));
        exit(1);
    }

    uint64_t asq = (uint64_t)admin_queue_ptr;
    uint64_t acq = (uint64_t)admin_queue_ptr + AEOLUS_HOST_PGSIZE;
    *(uint32_t *)(reg_ptr_uint + NVME_REG_AQA) = ((AEOLUS_ADMIN_QUEUE_DEPTH - 1) << 16) | (AEOLUS_ADMIN_QUEUE_DEPTH - 1);
    *(uint64_t *)(reg_ptr_uint + NVME_REG_ASQ) = admin_queue_phys_addr[0];
    *(uint64_t *)(reg_ptr_uint + NVME_REG_ACQ) = admin_queue_phys_addr[1];
    // AEOLUS_LOG_INFO("Admin queue phy addr: 0x%lx, 0x%lx", admin_queue_phys_addr[0], admin_queue_phys_addr[1]);

    admin_qp = new AdminQueuePair(
        (volatile uint32_t *)asq, 
        (volatile uint32_t *)acq, 
        NVME_BROADCAST_NSID, 
        (uint32_t *)(reg_ptr_uint + NVME_REG_SQTDBL), 
        (uint32_t *)(reg_ptr_uint + NVME_REG_CQHDBL), 
        AEOLUS_ADMIN_QUEUE_DEPTH
    );
    AEOLUS_LOG_INFO("Set admin_qp queue attributes done.");

    // Enable controller.
    *(uint32_t *)(reg_ptr_uint + NVME_REG_CC) |= NVME_REG_CC_EN;
    while (!(*(uint32_t volatile *)(reg_ptr_uint + NVME_REG_CSTS) & NVME_REG_CSTS_RDY))
        ;
    AEOLUS_LOG_INFO("Enable controller done.");

    // Set number of I/O queues. We will tentatively set a large number to the queue number
    // and then run the get-feature command so as to get the largest queue number supported.

    uint32_t status = admin_qp->set_num_queues(0xfffe, 0xfffe);   // Maximum queue pairs supported by NVMe.
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("Set number of queues failed with status 0x%x", status);
        exit(1);
    }
    AEOLUS_LOG_INFO("Set number of queues done.");
    
    uint16_t max_sq_num, max_cq_num;
    status = admin_qp->get_num_queues(max_sq_num, max_cq_num);
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("Get number of queues failed with status 0x%x", status);
        exit(1);
    }
    max_queue_num = MIN(max_sq_num, max_cq_num);
    AEOLUS_LOG_INFO("Maximum queue number supported: %d.", max_queue_num);

    // Decide the namespace to use. The namespace with the lowest number will be chosen.

    void *temp_buffer;
    uint64_t *temp_buffer_phys_addr;
    alloc_host_memory(&temp_buffer, AEOLUS_HOST_PGSIZE, &temp_buffer_phys_addr);
    status = admin_qp->identify(0x02, 0x0, 0, temp_buffer_phys_addr[0]);
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("Get namespace list failed with status 0x%x", status);
        exit(1);
    }
    active_ns = *((uint32_t *)temp_buffer);

    // Get device capacity.

    status = admin_qp->identify(0x00, 0x0, active_ns, temp_buffer_phys_addr[0]);
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("Get namespace structure 0x%x", status);
        exit(1);
    }
    max_lb_num = *((uint64_t *)temp_buffer);
    AEOLUS_LOG_INFO("Active ns: %d, Maximum logical block number supported: %lu.", active_ns, max_lb_num);

    // Get maximum IO size.

    status = admin_qp->identify(0x01, 0x0, 0x0, temp_buffer_phys_addr[0]);
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("Get controller structure failed with status 0x%x", status);
        exit(1);
    }
    max_io_size = *((uint8_t *)((uint64_t)temp_buffer + 77));
    max_io_size = AEOLUS_HOST_PGSIZE * (1 << max_io_size);
    AEOLUS_LOG_INFO("Maximum IO size supported: %d B.", max_io_size);
    free_host_memory(temp_buffer, temp_buffer_phys_addr);

    // Get free queue pair IDs.
    for (int i=1; i<max_queue_num; i++)
    {
        free_qps.push_back(i);
    }
}

Device::~Device()
{
    AEOLUS_LOG_INFO("Closing device %d", ssd_id);
    free_host_memory(admin_queue_ptr, admin_queue_phys_addr);
    AEOLUS_CUDA_CHECK(cudaHostUnregister(reg_ptr));
    munmap(reg_ptr, NVME_BAR0_SIZE);
    close(device_fd);
}

int Device::alloc_host_memory(void **ptr, uint64_t size, uint64_t** phys_addr)
{
    posix_memalign(ptr, AEOLUS_HOST_PGSIZE, size);
    memset(*ptr, 0, size);
    size_t num_page = CEIL(size, AEOLUS_HOST_PGSIZE);
    *phys_addr      = (uint64_t*)malloc(sizeof(uint64_t) * num_page);
    nvm_ioctl_map req;  // Request physical address.
    req.vaddr_start = (uint64_t)*ptr;
    req.n_pages     = num_page;
    req.ioaddrs     = *phys_addr;

    return ioctl(device_fd, NVM_MAP_HOST_MEMORY, &req);
}

void Device::free_host_memory(void *ptr, uint64_t* phys_addr)
{
    ioctl(device_fd, NVM_UNMAP_MEMORY, (uint64_t)ptr);
    free(phys_addr);
    free(ptr);
}

int Device::alloc_device_memory(void **ptr, aeolus_dev_mem_context** context, uint64_t size, uint64_t** phys_addr)
{
    *context = (aeolus_dev_mem_context*)malloc(sizeof(aeolus_dev_mem_context));
    size = size / AEOLUS_DEVICE_PGSIZE * AEOLUS_DEVICE_PGSIZE + AEOLUS_DEVICE_PGSIZE;
    AEOLUS_CUDA_CHECK(cudaMalloc(&((*context)->ptr), size + AEOLUS_DEVICE_PGSIZE));
    *ptr = (void *)((uint64_t)((*context)->ptr) / AEOLUS_DEVICE_PGSIZE * AEOLUS_DEVICE_PGSIZE + AEOLUS_DEVICE_PGSIZE);
    int flag = 0;
    if ((uint64_t)*ptr != (uint64_t)((*context)->ptr))
    {
        flag = 1;
    }
    (*context)->ioaddrs = malloc(sizeof(uint64_t) * (size / AEOLUS_DEVICE_PGSIZE + flag));
    *phys_addr = (uint64_t*)(*context)->ioaddrs;
    nvm_ioctl_map req;
    req.vaddr_start = (uint64_t)((*context)->ptr);
    req.n_pages     = size / AEOLUS_DEVICE_PGSIZE + flag;
    req.ioaddrs     = *phys_addr;
    *phys_addr      += flag;
    
    return ioctl(device_fd, NVM_MAP_DEVICE_MEMORY, &req);
}

void Device::free_device_memory(aeolus_dev_mem_context* context)
{
    ioctl(device_fd, NVM_UNMAP_MEMORY, (uint64_t)(context->ptr));
    free(context->ioaddrs);
    AEOLUS_CUDA_CHECK(cudaFree(context->ptr));
    free(context);
}