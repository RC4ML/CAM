#include "controller.cuh"
#include <algorithm>

Controller::Controller(
    std::vector<Device *> ssd_list, int32_t num_queue_per_ssd, int32_t max_io_size,
    int32_t queue_depth, aeolus_dist_type dist_type, aeolus_buf_type buf_type,
    uint64_t *pinned_buf_phys, uint64_t pinned_buf_size
)
{
    // Check if the input parameters are valid.

    AEOLUS_CUDA_CHECK(cudaGetDevice(&gpu_id));
    ssd_count = ssd_list.size();
    if (ssd_count <= 0)
    {{
        AEOLUS_LOG_ERROR("Empty SSD list delivered to Controller.");
        exit(-1);
    }}
    // Get maximum queue number and IO size of SSDs.
    max_queue_num   = INT_MAX;
    max_trans_size  = INT_MAX;
    for (auto ssd : ssd_list)
    {
        max_queue_num   = MIN(max_queue_num, ssd->free_qps.size());
        max_trans_size  = MIN(max_trans_size, ssd->max_io_size);
    }
    if (num_queue_per_ssd < 0)
    {
        num_queue_per_ssd = max_queue_num + 1 + num_queue_per_ssd;
    }
    if (num_queue_per_ssd <= 0 || num_queue_per_ssd > max_queue_num)
    {
        AEOLUS_LOG_ERROR(
            "Invalid queue number per SSD delivered to Controller."
            "The range should be between 1 and %d.", max_queue_num
        );
        exit(-1);
    }
    if (max_io_size == AEOLUS_MAX_DATA_TRANSFER)
    {
        // Less than 2 MiB IO size is to ensure the PRP list of a request won't exceed a page.
        max_io_size = MIN(max_trans_size, 2*1024*1024); 
    }
    if (max_io_size < 512 || max_io_size > max_trans_size || max_io_size > 2*1024*1024)
    {
        AEOLUS_LOG_ERROR(
            "Invalid max io size delivered to Controller."
            "The range should be between 512 and %d", MIN(max_trans_size, 2*1024*1024)
        );
        exit(-1);
    }
    if (!isPowerOfTwo(max_io_size))
    {
        AEOLUS_LOG_ERROR(
            "Invalid max io size delivered to Controller."
            "The value should be power of 2."
        );
        exit(-1);
    }

    this->ssd_list          = ssd_list;
    this->num_queue_per_ssd = num_queue_per_ssd;
    this->max_io_size       = max_io_size;
    this->queue_depth       = queue_depth;
    this->dist_type         = dist_type;
    this->buf_type          = buf_type;

    // Compute SSD LB prefix sum.

    h_ssd_num_lbs = new uint64_t[ssd_count];
    for (int i = 0; i < ssd_count; i++)
        h_ssd_num_lbs[i] = ssd_list[i]->max_lb_num;
    AEOLUS_CUDA_CHECK(cudaMalloc(&d_ssd_num_lbs, ssd_count * sizeof(uint64_t)));
    AEOLUS_CUDA_CHECK(cudaMemcpy(d_ssd_num_lbs, h_ssd_num_lbs, ssd_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Alloc shared buffers.
    
    AEOLUS_CUDA_CHECK(cudaMalloc(&ssd_num_reqs, ssd_count * sizeof(int)));
    if (dist_type != AEOLUS_DIST_STRIPE)
    {
        AEOLUS_LOG_ERROR("Controller only supports AEOLUS_DIST_STRIPE distribution type for now\n");
    }
    AEOLUS_CUDA_CHECK(cudaMalloc(&distributed_reqs, AEOLUS_MAX_NUM_REQUESTS * sizeof(Request)));
    AEOLUS_CUDA_CHECK(cudaMalloc(&req_ids, ssd_count * sizeof(int)));

    // Create SSD IO queue pairs.

    qpid_list = new int32_t *[ssd_count];
    for (int i=0; i<ssd_count; i++)
    {
        qpid_list[i] = new int32_t[num_queue_per_ssd];
        for (int j=0; j<num_queue_per_ssd; j++)
        {
            qpid_list[i][j] = ssd_list[i]->free_qps[0];
            ssd_list[i]->free_qps.erase(ssd_list[i]->free_qps.begin());
        }
    }

    int sq_size = MAX(AEOLUS_HOST_PGSIZE, queue_depth*NVME_SQ_ENTRY_SIZE);
    assert((sq_size % AEOLUS_HOST_PGSIZE) == 0);

    void *d_qp_ptr;
    int ret = ssd_list[0]->alloc_device_memory(&d_qp_ptr, &qp_ctx, 2*sq_size*ssd_count*num_queue_per_ssd, &qp_phys);
    if (ret != 0)
    {
        AEOLUS_LOG_ERROR("Failed to allocate device memory for SSD IO queues: %s", strerror(ret));
        exit(-1);
    }
    AEOLUS_CUDA_CHECK(cudaMemset(d_qp_ptr, 0, 2*sq_size*ssd_count*num_queue_per_ssd));

    AEOLUS_CUDA_CHECK(cudaMalloc(&d_ssdqp, ssd_count*num_queue_per_ssd*sizeof(IoQueuePair)));

    for (int i=0; i<ssd_count; i++)
    {
        for (int j=0; j<num_queue_per_ssd; j++)
        {
            uint64_t sq_virt = (uint64_t)d_qp_ptr + sq_size * (2*i*num_queue_per_ssd+2*j);
            uint64_t cq_virt = (uint64_t)d_qp_ptr + sq_size * (2*i*num_queue_per_ssd+2*j+1);

            int qid = qpid_list[i][j];

            // Create CQ.
            int offset = sq_size * (2*i*num_queue_per_ssd+2*j+1);
            uint64_t cq_phys = qp_phys[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
            ret = ssd_list[i]->admin_qp->create_cq_cont(qid, cq_phys, queue_depth);
            if (ret != 0)
            {
                AEOLUS_LOG_ERROR(
                    "Failed to create CQ %d for SSD %d with status 0x%x", 
                    qid, i, ret
                );
                exit(-1);
            }

            // Create SQ.
            offset = sq_size * (2*i*num_queue_per_ssd+2*j);
            uint64_t sq_phys = qp_phys[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
            ret = ssd_list[i]->admin_qp->create_sq_cont(qid, qid, sq_phys, queue_depth);
            if (ret != 0)
            {
                AEOLUS_LOG_ERROR(
                    "Failed to create SQ %d for SSD %d with status 0x%x", 
                    qid, i, ret
                );
                exit(-1);
            }
            // AEOLUS_LOG_INFO("CQ phy addr: 0x%lx, SQ phy addr: 0x%lx", cq_phys, sq_phys);

            // Create auxiliary data structures.
            uint32_t *d_cmd_id_to_req_id;
            AEOLUS_CUDA_CHECK(cudaMalloc(&d_cmd_id_to_req_id, sizeof(uint32_t)*queue_depth));
            uint32_t *d_cmd_id_to_sq_pos;
            AEOLUS_CUDA_CHECK(cudaMalloc(&d_cmd_id_to_sq_pos, sizeof(uint32_t)*queue_depth));
            bool *d_sq_entry_busy;
            AEOLUS_CUDA_CHECK(cudaMalloc(&d_sq_entry_busy, 1*queue_depth));
            AEOLUS_CUDA_CHECK(cudaMemset(d_sq_entry_busy, 0, 1*queue_depth));
            IoQueuePair h_ssdqp(
                (volatile uint32_t *)sq_virt, (volatile uint32_t *)cq_virt,
                ssd_list[i]->active_ns, 
                (uint32_t *)((uint64_t)ssd_list[i]->reg_ptr + NVME_REG_SQTDBL + qid * NVME_DBLSTRIDE),
                (uint32_t *)((uint64_t)ssd_list[i]->reg_ptr + NVME_REG_CQHDBL + qid * NVME_DBLSTRIDE),
                queue_depth, d_cmd_id_to_req_id, d_cmd_id_to_sq_pos, d_sq_entry_busy
            );
            // AEOLUS_LOG_INFO("Created SSD IO queue pair %d for SSD %d.", qid, i);
            AEOLUS_CUDA_CHECK(cudaMemcpy(
                d_ssdqp + i*num_queue_per_ssd+j, &h_ssdqp, 
                sizeof(IoQueuePair), cudaMemcpyHostToDevice
            ));
        }
    }

    uint64_t io_buf_size = (uint64_t)max_io_size*ssd_count*num_queue_per_ssd*queue_depth;
    uint64_t *h_iobuf_phys;
    if (buf_type == AEOLUS_BUF_USER) {
        
        // Allocate IO buffer.
        
        AEOLUS_LOG_INFO("Allocating IO buffer.");
        int ret = ssd_list[0]->alloc_device_memory(
            &d_iobuf_ptr, &iobuf_ctx, io_buf_size, &h_iobuf_phys
        );
        if (ret != 0)
        {
            AEOLUS_LOG_ERROR("Failed to allocate device memory for IO buffer: %s", strerror(ret));
            exit(-1);
        }
    } else 
    {
        h_iobuf_phys = pinned_buf_phys;
        io_buf_size = pinned_buf_size;
    }
    if (h_iobuf_phys != nullptr)
    {
        AEOLUS_CUDA_CHECK(cudaMalloc(&d_iobuf_phys, sizeof(uint64_t)));
        AEOLUS_CUDA_CHECK(cudaMemcpy(d_iobuf_phys, h_iobuf_phys, sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // Allocate PRP list.
    
    if (max_io_size > AEOLUS_HOST_PGSIZE * 2) {
        uint64_t prp_list_size = io_buf_size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t);
        if (io_buf_size > 0)
        {
            AEOLUS_LOG_INFO("Allocating PRP buffer.");
            ssd_list[0]->alloc_host_memory((void **)&prp_list, prp_list_size, &h_prp_phys);

            // Fill in PRP table.
            for (int i = 0; i < io_buf_size / AEOLUS_DEVICE_PGSIZE; i++)
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
        }

        // Move PRP physical address to GPU.
        size_t prp_phys_size = CEIL(prp_list_size, AEOLUS_HOST_PGSIZE) * sizeof(uint64_t);
        AEOLUS_CUDA_CHECK(cudaMalloc(&d_prp_phys, prp_phys_size));
        AEOLUS_CUDA_CHECK(cudaMemcpy(d_prp_phys, h_prp_phys, prp_phys_size, cudaMemcpyHostToDevice));
    }
}

Controller::~Controller()
{
    AEOLUS_LOG_INFO("Cleaning up controller.");
    if (buf_type == AEOLUS_BUF_USER) {
        if (max_io_size > 8192) {
            AEOLUS_CUDA_CHECK(cudaFree(d_prp_phys));
            ssd_list[0]->free_host_memory(prp_list, h_prp_phys);
        }
        ssd_list[0]->free_device_memory(iobuf_ctx);
        AEOLUS_CUDA_CHECK(cudaFree(d_iobuf_phys));
    }

    IoQueuePair *h_ssdqp = (IoQueuePair *)malloc(sizeof(IoQueuePair));
    for (int i=0; i<ssd_count; i++)
    {
        for (int j=0; j<num_queue_per_ssd; j++)
        {
            AEOLUS_CUDA_CHECK(cudaMemcpy(
                h_ssdqp, d_ssdqp + i*num_queue_per_ssd+j, 
                sizeof(IoQueuePair), cudaMemcpyDeviceToHost
            ));
            AEOLUS_CUDA_CHECK(cudaFree(h_ssdqp->cmd_id_to_req_id)); 
            AEOLUS_CUDA_CHECK(cudaFree(h_ssdqp->cmd_id_to_sq_pos));
            AEOLUS_CUDA_CHECK(cudaFree(h_ssdqp->sq_entry_busy));
            ssd_list[i]->admin_qp->delete_sq(qpid_list[i][j]);
            ssd_list[i]->admin_qp->delete_cq(qpid_list[i][j]);
            ssd_list[i]->free_qps.push_back(qpid_list[i][j]);
        }
        delete [] qpid_list[i];
    }
    delete [] qpid_list;

    AEOLUS_CUDA_CHECK(cudaFree(d_ssdqp));
    ssd_list[0]->free_device_memory(qp_ctx);
    free(h_ssdqp);
    delete [] h_ssd_num_lbs;
    AEOLUS_CUDA_CHECK(cudaFree(d_ssd_num_lbs));
    AEOLUS_LOG_INFO("Cleaning up controller done.");
}

__global__ static void rw_data_kernel(uint32_t opcode, int ssd_id, uint64_t start_lb, uint64_t num_lb, int num_queues_per_ssd, IoQueuePair *ssdqp, uint64_t *prp1, uint64_t *prp2, int queue_depth, int max_io_size, aeolus_buf_type buf_type)
{
    uint32_t cid;
    int global_queue_id = ssd_id * num_queues_per_ssd;
    uint64_t global_pos = (uint64_t)global_queue_id * queue_depth;
    uint64_t io_addr;
    if (buf_type == AEOLUS_BUF_USER)
        io_addr = prp1[0] + global_pos * max_io_size; // assume contiguous!
    else
    {
        io_addr = prp1[0];
        global_pos = 0;
    }
    uint64_t io_addr2 = io_addr / AEOLUS_HOST_PGSIZE * AEOLUS_HOST_PGSIZE + AEOLUS_HOST_PGSIZE;
    if (num_lb * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
    {
        int prp_size = max_io_size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t); // PRP list size of a request
        uint64_t offset = global_pos * prp_size;
        io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
    }
    ssdqp[global_queue_id].submit(cid, opcode, io_addr, io_addr2, start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, NVME_RW_LIMITED_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[global_queue_id].poll(status, cid);
    // printf("ssd_id: %d, start_lb: %lu, cmd_id: %u\n", ssd_id, start_lb, ssdqp[global_queue_id].cmd_id);
    if (status != 0)
    {
        AEOLUS_LOG_ERROR("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

void Controller::lb_to_ssd_id(uint64_t lb, int &ssd_id, uint64_t &local_lb)
{
    int lbs_per_max_io_size = max_io_size / AEOLUS_LB_SIZE;
    if (lb % lbs_per_max_io_size != 0)
    {
        AEOLUS_LOG_ERROR("Unaligned start LB %lu is unsupported now", lb);
        exit(-1);
    }
    ssd_id = lb / lbs_per_max_io_size % ssd_count;
    local_lb = lb / lbs_per_max_io_size / ssd_count * lbs_per_max_io_size;
    if (local_lb >= h_ssd_num_lbs[ssd_id])
    {
        AEOLUS_LOG_ERROR("Out of bound start LB %lu", lb);
        exit(-1);
    }
}

void Controller::read_data(uint64_t start_lb, uint64_t num_lb, void *buf)
{
    int ssd_id;
    uint64_t local_lb;
    lb_to_ssd_id(start_lb, ssd_id, local_lb);
    rw_data_kernel<<<1, 1>>>(
        NVME_OPCODE_READ, ssd_id, local_lb, num_lb, num_queue_per_ssd, 
        d_ssdqp, d_iobuf_phys, d_prp_phys, queue_depth, max_io_size, buf_type
    );
    if (buf_type == AEOLUS_BUF_USER) {
        AEOLUS_CUDA_CHECK(cudaMemcpy(
            buf, (uint8_t *)d_iobuf_ptr + (uint64_t)ssd_id * num_queue_per_ssd *
            queue_depth * max_io_size, 
            num_lb * AEOLUS_LB_SIZE, cudaMemcpyDeviceToHost
        ));
    } else {
        // TODO!
    }
}

void Controller::write_data(uint64_t start_lb, uint64_t num_lb, void *buf)
{
    int ssd_id;
    uint64_t local_lb;
    lb_to_ssd_id(start_lb, ssd_id, local_lb);
    if (buf_type == AEOLUS_BUF_USER) {
        AEOLUS_CUDA_CHECK(cudaMemcpy(
            (uint8_t *)d_iobuf_ptr + (uint64_t)ssd_id * num_queue_per_ssd *
            queue_depth * max_io_size, 
            buf, num_lb * AEOLUS_LB_SIZE, cudaMemcpyHostToDevice
        ));
    } else {
        // TODO!
    }
    rw_data_kernel<<<1, 1>>>(
        NVME_OPCODE_WRITE, ssd_id, local_lb, num_lb, num_queue_per_ssd, 
        d_ssdqp, d_iobuf_phys, d_prp_phys, queue_depth, max_io_size, buf_type
    );
    // AEOLUS_CUDA_CHECK(cudaDeviceSynchronize());
}