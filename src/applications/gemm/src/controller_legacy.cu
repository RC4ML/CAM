#include "controller.cuh"

__device__ static void lb_to_ssd_id(uint64_t lb, int num_ssds, uint64_t *ssd_num_lbs, int max_io_size, int &ssd_id, uint64_t &start_lb)
{
    int lbs_per_max_io_size = max_io_size / AEOLUS_LB_SIZE;
    assert(lb % lbs_per_max_io_size == 0);
    ssd_id = lb / lbs_per_max_io_size % num_ssds;
    start_lb = lb / lbs_per_max_io_size / num_ssds * lbs_per_max_io_size;
    assert(start_lb < ssd_num_lbs[ssd_id]);
}

__global__ static void do_read_req_kernel(Request *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, IoQueuePair *ssdqp, uint64_t *prp1, uint64_t *IO_buf_base, uint64_t *prp2, int queue_depth, int max_io_size, aeolus_buf_type buf_type)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / AEOLUS_WARP_SIZE; // global queue id
    int lane_id = thread_id % AEOLUS_WARP_SIZE;
    int ssd_id = warp_id / num_warps_per_ssd;
    if (ssd_id >= num_ssds)
        return;

    // submit first page of double buffer
    assert(thread_id < num_reqs);
    int base_req_id = thread_id - lane_id;
    int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % queue_depth;

    uint64_t global_pos = (uint64_t)warp_id * queue_depth + sq_pos;
    uint64_t io_addr;
    uint64_t io_addr2;
    if (buf_type == AEOLUS_BUF_USER)
    {
        io_addr = prp1[0] + global_pos * max_io_size; // assume contiguous!
        io_addr2 = io_addr / AEOLUS_HOST_PGSIZE * AEOLUS_HOST_PGSIZE + AEOLUS_HOST_PGSIZE;
    }
    else
    {
        io_addr = reqs[thread_id].dest_addr;
        io_addr2 = reqs[thread_id].next_addr;   // io_size <= 8KB
        global_pos = reqs[thread_id].next_addr; // io_size > 8KB
    }
    int prp_size = max_io_size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t); // PRP list size of a request
    if (reqs[thread_id].num_items * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
    {
        uint64_t offset = global_pos * prp_size;
        io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
    }
    if (lane_id == 0)
    {
        // ssdqp[warp_id].cmd_id = 0;
        // printf("queue %d cmd_id %d\n", warp_id, ssdqp[warp_id].cmd_id);
        for (int i = 0; i < queue_depth; i++)
            ssdqp[warp_id].sq_entry_busy[i] = false;
    }
    int num_lbs = reqs[thread_id].num_items ? reqs[thread_id].num_items - 1 : 0;
    ssdqp[warp_id].fill_sq(
        ssdqp[warp_id].cmd_id + lane_id,               // command id
        sq_pos,                                        // position in SQ
        NVME_OPCODE_READ,                              // opcode
        io_addr,                                       // prp1
        io_addr2,                                      // prp2
        reqs[thread_id].start_lb & 0xffffffff,         // start lb low
        (reqs[thread_id].start_lb >> 32) & 0xffffffff, // start lb high
        NVME_RW_LIMITED_RETRY_MASK | num_lbs,          // number of LBs
        thread_id                                      // req id
    );
    // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, thread_id, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

    __threadfence_system();
    // __syncwarp();
    if (lane_id == 0)
    {
        ssdqp[warp_id].cmd_id += AEOLUS_WARP_SIZE;
        ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + AEOLUS_WARP_SIZE) % queue_depth;
        // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
        *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
    }

    int stride = num_ssds * num_warps_per_ssd * AEOLUS_WARP_SIZE;
    for (int i = thread_id + stride; i < num_reqs + stride; i += stride)
    {
        int prev_sq_tail = ssdqp[warp_id].sq_tail;
        base_req_id = i - lane_id; // first req_id in warp
        if (i < num_reqs)
        {
            // submit second page of double buffer
            int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % queue_depth;

            uint64_t global_pos = (uint64_t)warp_id * queue_depth + sq_pos;
            uint64_t io_addr;
            uint64_t io_addr2;
            if (buf_type == AEOLUS_BUF_USER)
            {
                io_addr = prp1[0] + global_pos * max_io_size; // assume contiguous!
                io_addr2 = io_addr / AEOLUS_HOST_PGSIZE * AEOLUS_HOST_PGSIZE + AEOLUS_HOST_PGSIZE;
            }
            else
            {
                io_addr = reqs[i].dest_addr;
                io_addr2 = reqs[i].next_addr;   // io_size <= 8KB
                global_pos = reqs[i].next_addr; // io_size > 8KB
            }
            if (reqs[thread_id].num_items * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
            {
                uint64_t offset = global_pos * prp_size;
                io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
            }
            assert(ssdqp[warp_id].sq_entry_busy[sq_pos] == false);
            // if (i >= stride * 4 && !req_processed[i - stride * 4])
            // {
            //     printf("thread %d req_id %d not processed\n", thread_id, i - stride * 4);
            //     for (int i = 0; i < ssdqp[warp_id].cmd_id; i++)
            //     {
            //         int req_id = ssdqp[warp_id].cmd_id_to_req_id[i];
            //         int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[i];
            //         if (req_id != 0xffffffff)
            //             printf("thread %d cmd_id %d req_id %d processed %d sq_pos %d busy %d\n", thread_id, i, req_id, req_processed[req_id], sq_pos, ssdqp[warp_id].sq_entry_busy[sq_pos]);
            //     }
            //     assert(0);
            // }
            int num_lbs = reqs[i].num_items ? reqs[i].num_items - 1 : 0;
            ssdqp[warp_id].fill_sq(
                ssdqp[warp_id].cmd_id + lane_id,       // command id
                sq_pos,                                // position in SQ
                NVME_OPCODE_READ,                      // opcode
                io_addr,                               // prp1
                io_addr2,                              // prp2
                reqs[i].start_lb & 0xffffffff,         // start lb low
                (reqs[i].start_lb >> 32) & 0xffffffff, // start lb high
                NVME_RW_LIMITED_RETRY_MASK | num_lbs,  // number of LBs
                i                                      // req id
            );
            // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, i, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

            __threadfence_system();
            // __syncwarp();
            if (lane_id == 0)
            {
                int cnt = num_reqs - base_req_id < AEOLUS_WARP_SIZE ? num_reqs - base_req_id : AEOLUS_WARP_SIZE;
                assert(cnt == AEOLUS_WARP_SIZE);
                ssdqp[warp_id].cmd_id += cnt;
                ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + cnt) % queue_depth;
                // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
                *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
            }
        }

        // poll and copy the *previous* page of double buffer
        int prev_cq_head = ssdqp[warp_id].cq_head;
        if (lane_id == 0)
        {
            uint32_t code;
            ssdqp[warp_id].poll_range(code, prev_sq_tail, i < num_reqs);
            assert(code == 0);
            if (i + stride < num_reqs)
            {
                base_req_id += stride;
                int next_cnt = num_reqs - base_req_id < AEOLUS_WARP_SIZE ? num_reqs - base_req_id : AEOLUS_WARP_SIZE;
                for (int j = 0; j < next_cnt; j++)
                {
                    int sq_pos = (ssdqp[warp_id].sq_tail + j) % queue_depth;
                    if (ssdqp[warp_id].sq_entry_busy[sq_pos])
                    {
                        ssdqp[warp_id].poll_until_sq_entry_free(code, sq_pos);
                        assert(code == 0);
                    }
                }
            }
        }

        if (buf_type == AEOLUS_BUF_USER)
        {
            // copy data from IO buffer to app buffer
            for (int j = prev_cq_head; j != ssdqp[warp_id].cq_head; j = (j + 1) % queue_depth)
            {
                int cmd_id = (ssdqp[warp_id].cq[j * 4 + 3] & NVME_ENTRY_CID_MASK) % queue_depth;
                int req_id = ssdqp[warp_id].cmd_id_to_req_id[cmd_id];
                int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[cmd_id];
                for (int k = lane_id; k < reqs[req_id].num_items * AEOLUS_LB_SIZE / 8; k += AEOLUS_WARP_SIZE)
                    ((uint64_t *)reqs[req_id].dest_addr)[k] = IO_buf_base[(uint64_t)warp_id * queue_depth * max_io_size / 8 + sq_pos * max_io_size / 8 + k];
            }
        }
    }
}

__global__ static void do_write_req_kernel(Request *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, IoQueuePair *ssdqp, uint64_t *prp1, uint64_t *IO_buf_base, uint64_t *prp2, int queue_depth, int max_io_size, aeolus_buf_type buf_type)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / AEOLUS_WARP_SIZE; // global queue id
    int lane_id = thread_id % AEOLUS_WARP_SIZE;
    int ssd_id = warp_id / num_warps_per_ssd;
    if (ssd_id >= num_ssds)
        return;

    // submit first page of double buffer
    assert(thread_id < num_reqs);
    int base_req_id = thread_id - lane_id;
    int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % queue_depth;

    uint64_t global_pos = (uint64_t)warp_id * queue_depth + sq_pos;
    uint64_t io_addr;
    uint64_t io_addr2;
    if (buf_type == AEOLUS_BUF_USER)
    {
        io_addr = prp1[0] + global_pos * max_io_size; // assume contiguous!
        io_addr2 = io_addr / AEOLUS_HOST_PGSIZE * AEOLUS_HOST_PGSIZE + AEOLUS_HOST_PGSIZE;
    }
    else
    {
        io_addr = reqs[thread_id].dest_addr;
        io_addr2 = reqs[thread_id].next_addr;   // io_size <= 8KB
        global_pos = reqs[thread_id].next_addr; // io_size > 8KB
    }
    int prp_size = max_io_size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t); // PRP list size of a request
    if (reqs[thread_id].num_items * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
    {
        uint64_t offset = global_pos * prp_size;
        io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
    }
    if (lane_id == 0)
    {
        // ssdqp[warp_id].cmd_id = 0;
        // printf("queue %d cmd_id %d\n", warp_id, ssdqp[warp_id].cmd_id);
        for (int i = 0; i < queue_depth; i++)
            ssdqp[warp_id].sq_entry_busy[i] = false;
    }
    int num_lbs = reqs[thread_id].num_items ? reqs[thread_id].num_items - 1 : 0;
    ssdqp[warp_id].fill_sq(
        ssdqp[warp_id].cmd_id + lane_id,               // command id
        sq_pos,                                        // position in SQ
        NVME_OPCODE_WRITE,                             // opcode
        io_addr,                                       // prp1
        io_addr2,                                      // prp2
        reqs[thread_id].start_lb & 0xffffffff,         // start lb low
        (reqs[thread_id].start_lb >> 32) & 0xffffffff, // start lb high
        NVME_RW_LIMITED_RETRY_MASK | num_lbs,          // number of LBs
        thread_id                                      // req id
    );
    // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, thread_id, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

    if (buf_type == AEOLUS_BUF_USER)
    {
        for (int i = base_req_id; i < base_req_id + AEOLUS_WARP_SIZE; i++)
            for (int j = lane_id; j < reqs[i].num_items * AEOLUS_LB_SIZE / 8; j += AEOLUS_WARP_SIZE)
            {
                int sq_pos = (ssdqp[warp_id].sq_tail + i - base_req_id) % queue_depth;
                IO_buf_base[(uint64_t)warp_id * queue_depth * max_io_size / 8 + sq_pos * max_io_size / 8 + j] = ((uint64_t *)reqs[i].dest_addr)[j];
            }
    }

    __threadfence_system();
    // __syncwarp();
    if (lane_id == 0)
    {
        ssdqp[warp_id].cmd_id += AEOLUS_WARP_SIZE;
        ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + AEOLUS_WARP_SIZE) % queue_depth;
        // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
        *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
    }

    int stride = num_ssds * num_warps_per_ssd * AEOLUS_WARP_SIZE;
    for (int i = thread_id + stride; i < num_reqs + stride; i += stride)
    {
        int prev_sq_tail = ssdqp[warp_id].sq_tail;
        base_req_id = i - lane_id; // first req_id in warp
        if (i < num_reqs)
        {
            // submit second page of double buffer
            int sq_pos = (ssdqp[warp_id].sq_tail + lane_id) % queue_depth;

            uint64_t global_pos = (uint64_t)warp_id * queue_depth + sq_pos;
            uint64_t io_addr;
            uint64_t io_addr2;
            if (buf_type == AEOLUS_BUF_USER)
            {
                io_addr = prp1[0] + global_pos * max_io_size; // assume contiguous!
                io_addr2 = io_addr / AEOLUS_HOST_PGSIZE * AEOLUS_HOST_PGSIZE + AEOLUS_HOST_PGSIZE;
            }
            else
            {
                io_addr = reqs[i].dest_addr;
                io_addr2 = reqs[i].next_addr;   // io_size <= 8KB
                global_pos = reqs[i].next_addr; // io_size > 8KB
            }
            if (reqs[thread_id].num_items * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
            {
                uint64_t offset = global_pos * prp_size;
                io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
            }
            assert(ssdqp[warp_id].sq_entry_busy[sq_pos] == false);
            // if (i >= stride * 4 && !req_processed[i - stride * 4])
            // {
            //     printf("thread %d req_id %d not processed\n", thread_id, i - stride * 4);
            //     for (int i = 0; i < ssdqp[warp_id].cmd_id; i++)
            //     {
            //         int req_id = ssdqp[warp_id].cmd_id_to_req_id[i];
            //         int sq_pos = ssdqp[warp_id].cmd_id_to_sq_pos[i];
            //         if (req_id != 0xffffffff)
            //             printf("thread %d cmd_id %d req_id %d processed %d sq_pos %d busy %d\n", thread_id, i, req_id, req_processed[req_id], sq_pos, ssdqp[warp_id].sq_entry_busy[sq_pos]);
            //     }
            //     assert(0);
            // }
            int num_lbs = reqs[i].num_items ? reqs[i].num_items - 1 : 0;
            ssdqp[warp_id].fill_sq(
                ssdqp[warp_id].cmd_id + lane_id,       // command id
                sq_pos,                                // position in SQ
                NVME_OPCODE_WRITE,                     // opcode
                io_addr,                               // prp1
                io_addr2,                              // prp2
                reqs[i].start_lb & 0xffffffff,         // start lb low
                (reqs[i].start_lb >> 32) & 0xffffffff, // start lb high
                NVME_RW_LIMITED_RETRY_MASK | num_lbs,  // number of LBs
                i                                      // req id
            );
            // printf("thread %d req_id %d cmd_id %d num_completed %d sq_pos %d\n", thread_id, i, ssdqp[warp_id].cmd_id + lane_id, ssdqp[warp_id].num_completed, sq_pos);

            if (buf_type == AEOLUS_BUF_USER)
            {
                for (int j = base_req_id; j < base_req_id + AEOLUS_WARP_SIZE; j++)
                    for (int k = lane_id; k < reqs[j].num_items * AEOLUS_LB_SIZE / 8; k += AEOLUS_WARP_SIZE)
                    {
                        int sq_pos = (ssdqp[warp_id].sq_tail + j - base_req_id) % queue_depth;
                        IO_buf_base[(uint64_t)warp_id * queue_depth * max_io_size / 8 + sq_pos * max_io_size / 8 + k] = ((uint64_t *)reqs[j].dest_addr)[k];
                    }
            }

            __threadfence_system();
            // __syncwarp();
            if (lane_id == 0)
            {
                int cnt = num_reqs - base_req_id < AEOLUS_WARP_SIZE ? num_reqs - base_req_id : AEOLUS_WARP_SIZE;
                assert(cnt == AEOLUS_WARP_SIZE);
                ssdqp[warp_id].cmd_id += cnt;
                ssdqp[warp_id].sq_tail = (ssdqp[warp_id].sq_tail + cnt) % queue_depth;
                // printf("Warp %d, sq_tail is %p, set sqtdbl to %d\n", warp_id, ssdqp[warp_id].sqtdbl, ssdqp[warp_id].sq_tail);
                *ssdqp[warp_id].sqtdbl = ssdqp[warp_id].sq_tail;
            }
        }

        // poll and copy the *previous* page of double buffer
        if (lane_id == 0)
        {
            uint32_t code;
            ssdqp[warp_id].poll_range(code, prev_sq_tail, i < num_reqs);
            assert(code == 0);
            if (i + stride < num_reqs)
            {
                base_req_id += stride;
                int next_cnt = num_reqs - base_req_id < AEOLUS_WARP_SIZE ? num_reqs - base_req_id : AEOLUS_WARP_SIZE;
                for (int j = 0; j < next_cnt; j++)
                {
                    int sq_pos = (ssdqp[warp_id].sq_tail + j) % queue_depth;
                    if (ssdqp[warp_id].sq_entry_busy[sq_pos])
                    {
                        ssdqp[warp_id].poll_until_sq_entry_free(code, sq_pos);
                        assert(code == 0);
                    }
                }
            }
        }
    }
}

__global__ static void preprocess_io_req_1(Request *reqs, int num_reqs, int num_ssds, int *ssd_num_reqs, uint64_t *ssd_num_lbs, int max_io_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs; i += num_threads)
    {
        int ssd_id;
        uint64_t start_lb;  // Not used.
        lb_to_ssd_id(reqs[i].start_lb, num_ssds, ssd_num_lbs, max_io_size, ssd_id, start_lb);
        assert(ssd_id < num_ssds);
        atomicAdd(&ssd_num_reqs[ssd_id], 1);
    }
}

__global__ static void preprocess_io_req_2(Request *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, int *ssd_num_reqs, int *num_distributed_reqs)
{
    int max_bucket = 0;
    for (int i = 0; i < num_ssds; i++)
        if (ssd_num_reqs[i] > max_bucket)
            max_bucket = ssd_num_reqs[i];
    int num_reqs_per_chunk = num_warps_per_ssd * AEOLUS_WARP_SIZE;
    max_bucket = (max_bucket + num_reqs_per_chunk - 1) / num_reqs_per_chunk * num_reqs_per_chunk;
    *num_distributed_reqs = max_bucket * num_ssds;
}

__global__ static void distribute_io_req_1(int num_ssds, int num_warps_per_ssd, int *req_ids)
{
    int num_reqs_per_chunk = num_warps_per_ssd * AEOLUS_WARP_SIZE;
    for (int i = 0; i < num_ssds; i++)
        req_ids[i] = i * num_reqs_per_chunk;
}

__global__ static void distribute_io_req_2(Request *reqs, int num_reqs, int num_ssds, int num_warps_per_ssd, Request *distributed_reqs, int *req_ids, uint64_t *ssd_num_lbs, int max_io_size)
{
    int num_reqs_per_chunk = num_warps_per_ssd * AEOLUS_WARP_SIZE;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs; i += num_threads)
    {
        int ssd_id;
        uint64_t start_lb;
        lb_to_ssd_id(reqs[i].start_lb, num_ssds, ssd_num_lbs, max_io_size, ssd_id, start_lb);
        assert(ssd_id < num_ssds);
        int req_id;
        for (;;)
        {
            req_id = req_ids[ssd_id];
            int next_req_id = req_id + 1;
            if (next_req_id % num_reqs_per_chunk == 0)
                next_req_id += num_reqs_per_chunk * (num_ssds - 1);
            if (atomicCAS(&req_ids[ssd_id], req_id, next_req_id) == req_id)
                break;
        }
        distributed_reqs[req_id] = reqs[i];
        distributed_reqs[req_id].start_lb = start_lb;
    }
}

__global__ static void distribute_io_req_3(int num_ssds, int num_warps_per_ssd, Request *distributed_reqs, int *req_ids, int *num_distributed_reqs)
{
    int num_reqs_per_chunk = num_warps_per_ssd * AEOLUS_WARP_SIZE;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_ssds; i += num_threads)
        for (int j = req_ids[i]; j < *num_distributed_reqs;)
        {
            distributed_reqs[j].num_items = 0;
            distributed_reqs[j++].start_lb = 0;
            if (j % num_reqs_per_chunk == 0)
                j += num_reqs_per_chunk * (num_ssds - 1);
        }
}

void ControllerLegacy::submit_io_req(Request *req, int num_req, aeolus_access_dir dir, cudaStream_t stream, uint64_t* d_prp_phys) {
    if (num_req > AEOLUS_MAX_NUM_REQUESTS)
    {
        AEOLUS_LOG_ERROR("num_reqs %d > AEOLUS_MAX_NUM_REQUESTS %d", num_req, AEOLUS_MAX_NUM_REQUESTS);
        return;
    }
    AEOLUS_CUDA_CHECK(cudaMemsetAsync(ssd_num_reqs, 0, sizeof(int) * ssd_count, stream));
    int num_blocks = 8;
    preprocess_io_req_1<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(req, num_req, ssd_count, ssd_num_reqs, d_ssd_num_lbs, max_io_size);
    int *num_distributed_reqs;
    AEOLUS_CUDA_CHECK(cudaMalloc(&num_distributed_reqs, sizeof(int)));
    preprocess_io_req_2<<<1, 1, 0, stream>>>(req, num_req, ssd_count, num_queue_per_ssd, ssd_num_reqs, num_distributed_reqs);
    distribute_io_req_1<<<1, 1, 0, stream>>>(ssd_count, num_queue_per_ssd, req_ids);
    distribute_io_req_2<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(req, num_req, ssd_count, num_queue_per_ssd, distributed_reqs, req_ids, d_ssd_num_lbs, max_io_size);
    distribute_io_req_3<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(ssd_count, num_queue_per_ssd, distributed_reqs, req_ids, num_distributed_reqs);
    
    int h_num_distributed_reqs;
    AEOLUS_CUDA_CHECK(cudaMemcpy(&h_num_distributed_reqs, num_distributed_reqs, sizeof(int), cudaMemcpyDeviceToHost));
    int num_threads = ssd_count * num_queue_per_ssd * AEOLUS_WARP_SIZE;
    num_blocks = CEIL(num_threads, AEOLUS_NUM_THREADS_PER_BLOCK);
    if (d_prp_phys == nullptr)
        d_prp_phys = this->d_prp_phys;
    if (dir == AEOLUS_DIR_READ)
        do_read_req_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, h_num_distributed_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, d_iobuf_phys, (uint64_t *)d_iobuf_ptr, d_prp_phys, queue_depth, max_io_size, buf_type);
    else
        do_write_req_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, h_num_distributed_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, d_iobuf_phys, (uint64_t *)d_iobuf_ptr, d_prp_phys, queue_depth, max_io_size, buf_type);
    AEOLUS_CUDA_CHECK(cudaFree(num_distributed_reqs));
}