#include "controller.cuh"

__device__ static int req_id_to_ssd_id(int req_id, int num_ssds, int *ssd_num_reqs_prefix_sum)
{
    int ssd_id = 0;
    for (; ssd_id < num_ssds; ssd_id++)
        if (ssd_num_reqs_prefix_sum[ssd_id] > req_id)
            break;
    return ssd_id;
}

// Do NOT use std::pair in device function! Though this can be bypassed by --expt-relaxed-constexpr flag,
// it may contain bugs.
__device__ static void lb_to_ssd_id(uint64_t lb, int num_ssds, uint64_t *ssd_num_lbs, int max_io_size, int &ssd_id, uint64_t &start_lb)
{
    int lbs_per_max_io_size = max_io_size / AEOLUS_LB_SIZE;
    assert(lb % lbs_per_max_io_size == 0);
    ssd_id = lb / lbs_per_max_io_size % num_ssds;
    start_lb = lb / lbs_per_max_io_size / num_ssds * lbs_per_max_io_size;
    assert(start_lb < ssd_num_lbs[ssd_id]);
}

__global__ static void submit_io_req_kernel(Request *reqs, int num_reqs, int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, uint64_t *prp1, uint64_t *prp2, int *ssd_num_reqs_prefix_sum, int queue_depth, int max_io_size, uint32_t opcode, aeolus_buf_type buf_type)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        if (queue_id >= num_queues_per_ssd)
            printf("%d %d\n", queue_id, num_queues_per_ssd);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);
        int queue_pos = (ssdqp[global_queue_id].sq_tail + id_in_queue) % queue_depth;

        uint64_t global_pos = (uint64_t)global_queue_id * queue_depth + queue_pos;
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
        if (reqs[i].num_items * AEOLUS_LB_SIZE > AEOLUS_HOST_PGSIZE * 2)
        {
            int prp_size = max_io_size / AEOLUS_HOST_PGSIZE * sizeof(uint64_t); // PRP list size of a request
            uint64_t offset = global_pos * prp_size;
            io_addr2 = prp2[offset / AEOLUS_HOST_PGSIZE] + offset % AEOLUS_HOST_PGSIZE;
        }
        ssdqp[global_queue_id].fill_sq(
            ssdqp[global_queue_id].cmd_id + id_in_queue, // command id
            queue_pos,                                   // position in SQ
            opcode,                                      // opcode
            io_addr,                                     // prp1
            io_addr2,                                    // prp2
            reqs[i].start_lb & 0xffffffff,               // start lb low
            (reqs[i].start_lb >> 32) & 0xffffffff,       // start lb high
            NVME_RW_LIMITED_RETRY_MASK | (reqs[i].num_items - 1),     // number of LBs
            i                                            // req id
        );
    }
}

__global__ static void ring_sq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int num_reqs, int queue_depth)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);

        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (queue_depth - 1);
            if (cnt > queue_depth - 1)
                cnt = queue_depth - 1;
            ssdqp[global_queue_id].cmd_id += cnt;
            ssdqp[global_queue_id].sq_tail = (ssdqp[global_queue_id].sq_tail + cnt) % queue_depth;
            // printf("thread %d ssd %d queue %d end req %d cnt %d\n", thread_id, ssd_id, queue_id, ssd_num_reqs_prefix_sum[ssd_id], cnt);
            *ssdqp[global_queue_id].sqtdbl = ssdqp[global_queue_id].sq_tail;
        }
    }
}

__global__ static void copy_io_req_kernel(Request *reqs, int num_reqs, int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, uint64_t *IO_buf_base, int *ssd_num_reqs_prefix_sum, int queue_depth, int max_io_size, aeolus_buf_type buf_type)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / AEOLUS_WARP_SIZE;
    int lane_id = thread_id % AEOLUS_WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / AEOLUS_WARP_SIZE;
    for (int i = warp_id; i < num_reqs; i += num_warps)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);
        int complete_id = ssdqp[global_queue_id].num_completed + id_in_queue;
        int queue_pos = complete_id % queue_depth;

        if (lane_id == 0)
        {
            // printf("polling req %d ssd %d queue %d complete_id %d queue_pos %d num_completed %d\n", i, ssd_id, queue_id, complete_id, queue_pos, ssdqp[global_queue_id].num_completed);
            uint32_t current_phase = (complete_id / queue_depth) & 1;
            while (((ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
                ;
            uint32_t status = ssdqp[global_queue_id].cq[queue_pos * 4 + 3];
            uint32_t cmd_id = status & NVME_ENTRY_CID_MASK;
            if ((status >> 17) & NVME_ENTRY_SC_MASK)
            {
                AEOLUS_LOG_ERROR("thread %d cq[%d] status: 0x%x, cid: %d\n", thread_id, queue_pos, (status >> 17) & NVME_ENTRY_SC_MASK, cmd_id);
                assert(0);
            }
        }

        if (buf_type == AEOLUS_BUF_USER)
        {
            int cmd_id = ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & NVME_ENTRY_CID_MASK;
            int req_id = ssdqp[global_queue_id].cmd_id_to_req_id[cmd_id % queue_depth];
            int sq_pos = ssdqp[global_queue_id].cmd_id_to_sq_pos[cmd_id % queue_depth];
            for (int j = lane_id; j < reqs[req_id].num_items * AEOLUS_LB_SIZE / 8; j += AEOLUS_WARP_SIZE)
                ((uint64_t *)reqs[req_id].dest_addr)[j] = IO_buf_base[(uint64_t)global_queue_id * queue_depth * max_io_size / 8 + sq_pos * max_io_size / 8 + j];
        }
    }
}

__global__ static void ring_cq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int num_reqs, int queue_depth)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);

        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (queue_depth - 1);
            if (cnt > queue_depth - 1)
                cnt = queue_depth - 1;
            ssdqp[global_queue_id].num_completed += cnt;
            ssdqp[global_queue_id].cq_head = (ssdqp[global_queue_id].cq_head + cnt) % queue_depth;
            *ssdqp[global_queue_id].cqhdbl = ssdqp[global_queue_id].cq_head;
            // printf("queue %d num_completed %d cq_head %d\n", global_queue_id, ssdqp[global_queue_id].num_completed, ssdqp[global_queue_id].cq_head);
        }
    }
}

__global__ static void copy_write_data_kernel(Request *reqs, int num_reqs, int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, uint64_t *IO_buf_base, int *ssd_num_reqs_prefix_sum, int queue_depth, int max_io_size)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / AEOLUS_WARP_SIZE;
    int lane_id = thread_id % AEOLUS_WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / AEOLUS_WARP_SIZE;
    for (int i = warp_id; i < num_reqs; i += num_warps)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);
        int queue_pos = (ssdqp[global_queue_id].sq_tail + id_in_queue) % queue_depth;

        for (int j = lane_id; j < reqs[i].num_items * AEOLUS_LB_SIZE / 8; j += AEOLUS_WARP_SIZE)
            IO_buf_base[(uint64_t)global_queue_id * queue_depth * max_io_size / 8 + queue_pos * max_io_size / 8 + j] = ((uint64_t *)reqs[i].dest_addr)[j];
    }
}

__global__ static void poll_write_req_kernel(Request *reqs, int num_reqs, int num_ssds, int num_queues_per_ssd, IoQueuePair *ssdqp, uint64_t *IO_buf_base, int *ssd_num_reqs_prefix_sum, int queue_depth, int max_io_size)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / AEOLUS_WARP_SIZE;
    int lane_id = thread_id % AEOLUS_WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / AEOLUS_WARP_SIZE;
    for (int i = warp_id; i < num_reqs; i += num_warps)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (queue_depth - 1);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (queue_depth - 1);
        int complete_id = ssdqp[global_queue_id].num_completed + id_in_queue;
        int queue_pos = complete_id % queue_depth;

        if (lane_id == 0)
        {
            // printf("polling req %d ssd %d queue %d complete_id %d queue_pos %d num_completed %d\n", i, ssd_id, queue_id, complete_id, queue_pos, ssdqp[global_queue_id].num_completed);
            uint32_t current_phase = (complete_id / queue_depth) & 1;
            while (((ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
                ;
            uint32_t status = ssdqp[global_queue_id].cq[queue_pos * 4 + 3];
            uint32_t cmd_id = status & NVME_ENTRY_CID_MASK;
            if ((status >> 17) & NVME_ENTRY_SC_MASK)
            {
                AEOLUS_LOG_ERROR("thread %d cq[%d] status: 0x%x, cid: %d\n", thread_id, queue_pos, (status >> 17) & NVME_ENTRY_SC_MASK, cmd_id);
                assert(0);
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
        // assert(ssd_id < num_ssds);
        if (ssd_id < num_ssds && ssd_id >= 0)
        {
            atomicAdd(&ssd_num_reqs[ssd_id], 1);
        }
        else
        {
            AEOLUS_LOG_ERROR("ssd_id out of bound: %d\n", ssd_id);
        }
    }
}

__global__ static void preprocess_io_req_2(int num_ssds, int num_queues_per_ssd, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int queue_depth)
{
    for (int i = 0; i < num_ssds; i++)
    {
        // assert(ssd_num_reqs[i] <= num_queues_per_ssd * (queue_depth - 1));
        if (ssd_num_reqs[i] > num_queues_per_ssd * (queue_depth - 1))
        {
            AEOLUS_LOG_ERROR("ssd_num_reqs[%d]: %d\n", i, ssd_num_reqs[i]);
        }
        ssd_num_reqs_prefix_sum[i] = ssd_num_reqs[i];
        if (i > 0)
            ssd_num_reqs_prefix_sum[i] += ssd_num_reqs_prefix_sum[i - 1];
    }
}

__global__ static void distribute_io_req_1(int num_ssds, int *ssd_num_reqs_prefix_sum, int *req_ids)
{
    for (int i = 0; i < num_ssds; i++)
        req_ids[i] = i ? ssd_num_reqs_prefix_sum[i - 1] : 0;
}

__global__ static void distribute_io_req_2(Request *reqs, int num_reqs, int num_ssds, Request *distributed_reqs, int *req_ids, uint64_t *ssd_num_lbs, int max_io_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs; i += num_threads)
    {
        int ssd_id;
        uint64_t start_lb;
        lb_to_ssd_id(reqs[i].start_lb, num_ssds, ssd_num_lbs, max_io_size, ssd_id, start_lb);
        // assert(ssd_id < num_ssds);
        if (ssd_id < num_ssds && ssd_id >= 0)
        {
            int req_id = atomicAdd(&req_ids[ssd_id], 1);
            distributed_reqs[req_id] = reqs[i];
            distributed_reqs[req_id].start_lb = start_lb;
        }
    }
}

__global__ static void distribute_io_req_3(int num_ssds, int *ssd_num_reqs_prefix_sum, int *req_ids)
{
    for (int i = 0; i < num_ssds; i++)
    {
        if (req_ids[i] != ssd_num_reqs_prefix_sum[i])
        {
            AEOLUS_LOG_ERROR("req id %d %d\n", req_ids[i], ssd_num_reqs_prefix_sum[i]);
        }
        // assert(req_ids[i] == ssd_num_reqs_prefix_sum[i]);
    }
}

void ControllerDecoupled::submit_io_req(Request *reqs, int num_reqs, aeolus_access_dir dir, cudaStream_t stream, uint64_t* d_prp_phys)
{
    if (num_reqs > AEOLUS_MAX_NUM_REQUESTS)
    {
        AEOLUS_LOG_ERROR("num_reqs %d > AEOLUS_MAX_NUM_REQUESTS %d", num_reqs, AEOLUS_MAX_NUM_REQUESTS);
        exit(1);
    }
    if (num_reqs > ssd_count * num_queue_per_ssd * queue_depth)
    {
        AEOLUS_LOG_ERROR("num_reqs %d > ssd_count %d * num_queue_per_ssd %d * queue_depth %d", num_reqs, ssd_count, num_queue_per_ssd, queue_depth);
        exit(1);
    }
    AEOLUS_CUDA_CHECK(cudaMemsetAsync(ssd_num_reqs, 0, sizeof(int) * ssd_count, stream));
    int num_blocks = 32;
    preprocess_io_req_1<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, ssd_count, ssd_num_reqs, d_ssd_num_lbs, max_io_size);
    preprocess_io_req_2<<<1, 1, 0, stream>>>(ssd_count, num_queue_per_ssd, ssd_num_reqs, ssd_num_reqs_prefix_sum, queue_depth);
    distribute_io_req_1<<<1, 1, 0, stream>>>(ssd_count, ssd_num_reqs_prefix_sum, req_ids);
    distribute_io_req_2<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, ssd_count, distributed_reqs, req_ids, d_ssd_num_lbs, max_io_size);
    distribute_io_req_3<<<1, 1, 0, stream>>>(ssd_count, ssd_num_reqs_prefix_sum, req_ids);
    uint32_t opcode = NVME_OPCODE_READ;
    if (dir == AEOLUS_DIR_WRITE)
    {
        opcode = NVME_OPCODE_WRITE;
        if (buf_type == AEOLUS_BUF_USER)
            copy_write_data_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, num_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, (uint64_t *)d_iobuf_ptr, ssd_num_reqs_prefix_sum, queue_depth, max_io_size);
    }
    if (d_prp_phys == nullptr)
        d_prp_phys = this->d_prp_phys;
    submit_io_req_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, num_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, d_iobuf_phys, d_prp_phys, ssd_num_reqs_prefix_sum, queue_depth, max_io_size, opcode, buf_type);
    ring_sq_doorbell_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(ssd_count, num_queue_per_ssd, d_ssdqp, ssd_num_reqs, ssd_num_reqs_prefix_sum, num_reqs, queue_depth);
    this->num_reqs = num_reqs;
    this->stream = stream;
    this->dir = dir;
}

void ControllerDecoupled::poll()
{
    int num_blocks = 32;
    if (dir == AEOLUS_DIR_READ)
        copy_io_req_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, num_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, (uint64_t *)d_iobuf_ptr, ssd_num_reqs_prefix_sum, queue_depth, max_io_size, buf_type);
    else
        poll_write_req_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs, num_reqs, ssd_count, num_queue_per_ssd, d_ssdqp, (uint64_t *)d_iobuf_ptr, ssd_num_reqs_prefix_sum, queue_depth, max_io_size);
    ring_cq_doorbell_kernel<<<num_blocks, AEOLUS_NUM_THREADS_PER_BLOCK, 0, stream>>>(ssd_count, num_queue_per_ssd, d_ssdqp, ssd_num_reqs, ssd_num_reqs_prefix_sum, num_reqs, queue_depth);
}