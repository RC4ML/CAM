#include "queue.cuh"

__host__ __device__ void QueuePair::submit(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12)
{
    fill_sq(cmd_id, sq_tail, opcode, prp1, prp2, dw10, dw11, dw12);
    sq_tail = (sq_tail + 1) % queue_depth;
    *sqtdbl = sq_tail;
    cid = cmd_id;
    cmd_id = (cmd_id + 1) & NVME_ENTRY_CID_MASK;
}

__device__ void QueuePair::submit_fence(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12)
{
    fill_sq(cmd_id, sq_tail, opcode, prp1, prp2, dw10, dw11, dw12);
    __threadfence_system();
    sq_tail = (sq_tail + 1) % queue_depth;
    *sqtdbl = sq_tail;
    cid = cmd_id;
    cmd_id = (cmd_id + 1) & NVME_ENTRY_CID_MASK;
}

__host__ __device__ void QueuePair::fill_sq(uint32_t cid, uint32_t pos, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12, uint32_t req_id)
{
    // if (req_id == 1152)
    //     printf("%lx %lx %x %x %x %x %x %x\n", prp1, prp2, dw10, dw11, dw12, opcode, cid, namespace_id);
    sq[pos * 16 + 0] = opcode | (cid << 16);
    sq[pos * 16 + 1] = namespace_id;
    sq[pos * 16 + 6] = prp1 & 0xffffffff;
    sq[pos * 16 + 7] = prp1 >> 32;
    sq[pos * 16 + 8] = prp2 & 0xffffffff;
    sq[pos * 16 + 9] = prp2 >> 32;
    sq[pos * 16 + 10] = dw10;
    sq[pos * 16 + 11] = dw11;
    sq[pos * 16 + 12] = dw12;
    if (cmd_id_to_req_id)
        cmd_id_to_req_id[cid % queue_depth] = req_id;
    if (cmd_id_to_sq_pos)
        cmd_id_to_sq_pos[cid % queue_depth] = pos;
    if (sq_entry_busy)
        sq_entry_busy[pos] = true;
}

__host__ __device__ void QueuePair::poll(uint32_t &code, uint32_t cid)
{
    uint32_t current_phase = ((cmd_id - 1) / queue_depth) & 1;
    uint32_t status = cq[cq_head * 4 + 3];
    while (((status & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
        status = cq[cq_head * 4 + 3];
    if ((status & NVME_ENTRY_CID_MASK) != cid)
    {
        AEOLUS_LOG_ERROR("expected cid: %d, actual cid: %d", cid, status & NVME_ENTRY_CID_MASK);
        assert(0);
    }
    code = (status >> 17) & NVME_ENTRY_SC_MASK;
    num_completed++;
    cq_head = (cq_head + 1) % queue_depth;
    *cqhdbl = cq_head;
}

__host__ __device__ void QueuePair::poll_with_dw0(uint32_t &code, uint32_t cid, uint32_t &dw0)
{
    uint32_t current_phase = ((cmd_id - 1) / queue_depth) & 1;
    uint32_t status = cq[cq_head * 4 + 3];
    while (((status & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
        status = cq[cq_head * 4 + 3];
    if ((status & NVME_ENTRY_CID_MASK) != cid)
    {
        AEOLUS_LOG_ERROR("expected cid: %d, actual cid: %d", cid, status & NVME_ENTRY_CID_MASK);
        assert(0);
    }
    code = (status >> 17) & NVME_ENTRY_SC_MASK;
    dw0 = cq[cq_head * 4];
    num_completed++;
    cq_head = (cq_head + 1) % queue_depth;
    *cqhdbl = cq_head;
}

__device__ void IoQueuePair::poll_range(uint32_t &code, int expected_sq_head, bool should_break)
{
    // printf("cmd_id: %d, size: %d, current_phase: %d\n", cmd_id, size, current_phase);
    int i;
    uint32_t last_sq_head = ~0U;
    // int last_num_completed = num_completed;
    // int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for (i = cq_head; (num_completed & NVME_ENTRY_CID_MASK) != (cmd_id & NVME_ENTRY_CID_MASK); i = (i + 1) % queue_depth)
    {
        uint32_t current_phase = (num_completed / queue_depth) & 1;
        uint32_t status = cq[i * 4 + 3];
        uint64_t start = clock64();
        while (((status & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
        {
            status = cq[i * 4 + 3];
            if (clock64() - start > 1000000000)
            {
                AEOLUS_LOG_ERROR("timeout sq_tail=%d, cq_head=%d, i=%d, num_completed=%d, cmd_id=%d\n", sq_tail, cq_head, i, num_completed, cmd_id);
                AEOLUS_LOG_ERROR("last_sq_head: %d, expected_sq_head: %d\n", last_sq_head, expected_sq_head);
                // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
                // if (thread_id)
                //     return 0;
                // for (int m = 0; m < queue_depth; m++)
                // {
                //     printf("SQE %d\n", m);
                //     for (int n = 0; n < 16; n++)
                //         printf("DW%2d, %08x\n", n, sq[m * 16 + n]);
                // }
                // for (int m = 0; m < queue_depth; m++)
                // {
                //     printf("CQE %d\n", m);
                //     for (int n = 0; n < 4; n++)
                //         printf("DW%2d, %08x\n", n, cq[m * 4 + n]);
                // }
                code = 1;
            }
        }
        int cmd_id = status & NVME_ENTRY_CID_MASK;
        int sq_pos = cmd_id_to_sq_pos[cmd_id % queue_depth];
        if ((status >> 17) & NVME_ENTRY_SC_MASK)
        {
            printf("cq[%d] status: 0x%x, cid: %d\n", i, (status >> 17) & NVME_ENTRY_SC_MASK, status & NVME_ENTRY_CID_MASK);
            int req_id = cmd_id_to_req_id[cmd_id % queue_depth];
            printf("req_id: %d, sq_pos: %d\n", req_id, sq_pos);
            // for (int i = 0; i < 16; i++)
            //     printf("%08x ", sq[sq_pos * 16 + i]);
            // printf("\n");
            code = (status >> 17) & NVME_ENTRY_SC_MASK;
        }
        last_sq_head = cq[i * 4 + 2] & NVME_ENTRY_SQ_HEAD_MASK;
        sq_entry_busy[sq_pos] = false;
        // printf("thread %d freed sq_pos %d\n", thread_id, sq_pos);
        num_completed++;
        if (should_break && ((cq[i * 4 + 2] & NVME_ENTRY_SQ_HEAD_MASK) - expected_sq_head + queue_depth) % queue_depth <= AEOLUS_WARP_SIZE)
        {
            // printf("cq[%d] sq_head: %d, expected_sq_head: %d\n", i, cq[i * 4 + 2] & SQ_HEAD_MASK, expected_sq_head);
            i = (i + 1) % queue_depth;
            // if (num_completed - last_num_completed > 64)
            //     printf("%d: %d completed\n", thread_id, num_completed - last_num_completed);
            break;
        }
    }
    if (i != cq_head)
    {
        cq_head = i;
        // printf("cq_head is %p, set cqhdbl to %d\n", cqhdbl, cq_head);
        *cqhdbl = cq_head;
    }
    code = 0;
}

__device__ void IoQueuePair::poll_multiple(uint32_t &code, int cnt)
{
    for (int i = 0; i < cnt; i++)
    {
        uint32_t current_phase = (num_completed / queue_depth) & 1;
        int pos = (cq_head + i) % queue_depth;
        uint32_t status = cq[pos * 4 + 3];
        while (((status & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
            status = cq[pos * 4 + 3];
        int cmd_id = status & NVME_ENTRY_CID_MASK;
        int sq_pos = cmd_id_to_sq_pos[cmd_id % queue_depth];
        if ((status >> 17) & NVME_ENTRY_SC_MASK)
        {
            printf("cq[%d] status: 0x%x, cid: %d\n", pos, (status >> 17) & NVME_ENTRY_SC_MASK, status & NVME_ENTRY_CID_MASK);
            code = (status >> 17) & NVME_ENTRY_SC_MASK;
        }
        sq_entry_busy[sq_pos] = false;
        num_completed++;
    }
    cq_head = (cq_head + cnt) % queue_depth;
    *cqhdbl = cq_head;
    code = 0;
}

__device__ void IoQueuePair::poll_until_sq_entry_free(uint32_t &code, int expected_sq_pos) {
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // int last_num_completed = num_completed;
    // printf("thread %d want to free sq_pos: %d num_completed %d cmd_id %d\n", thread_id, expected_sq_pos, num_completed, cmd_id);
    int i;
    for (i = cq_head; (num_completed & NVME_ENTRY_CID_MASK) != (cmd_id & NVME_ENTRY_CID_MASK); i = (i + 1) % queue_depth)
    {
        uint32_t current_phase = (num_completed / queue_depth) & 1;
        uint32_t status = cq[i * 4 + 3];
        while (((status & NVME_ENTRY_PHASE_MASK) >> 16) == current_phase)
            status = cq[i * 4 + 3];
        int cmd_id = status & NVME_ENTRY_CID_MASK;
        int sq_pos = cmd_id_to_sq_pos[cmd_id % queue_depth];
        if ((status >> 17) & NVME_ENTRY_SC_MASK)
        {
            printf("cq[%d] status: 0x%x, cid: %d\n", i, (status >> 17) & NVME_ENTRY_SC_MASK, status & NVME_ENTRY_CID_MASK);
            int req_id = cmd_id_to_req_id[cmd_id % queue_depth];
            printf("req_id: %d, sq_pos: %d\n", req_id, sq_pos);
            // for (int i = 0; i < 16; i++)
            //     printf("%08x ", sq[sq_pos * 16 + i]);
            // printf("\n");
            code = (status >> 17) & NVME_ENTRY_SC_MASK;
        }
        sq_entry_busy[sq_pos] = false;
        // printf("thread %d manually freed sq_pos %d\n", thread_id, sq_pos);
        num_completed++;
        if (sq_pos == expected_sq_pos)
        {
            cq_head = (i + 1) % queue_depth;
            // printf("cq_head is %p, set cqhdbl to %d\n", cqhdbl, cq_head);
            *cqhdbl = cq_head;
            // if (num_completed - last_num_completed > 64)
            //     printf("%d: %d completed\n", thread_id, num_completed - last_num_completed);
            code = 0;
        }
    }
    // printf("thread %d failed to free sq_pos %d\n", thread_id, expected_sq_pos);
    code = 1;
}

__host__ __device__ void AdminQueuePair::submit_with_ns(uint32_t &cid, uint32_t opcode, uint32_t nsid, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12)
{
    fill_sq_with_ns(cmd_id, sq_tail, opcode, nsid, prp1, prp2, dw10, dw11, dw12);
    sq_tail = (sq_tail + 1) % queue_depth;
    *sqtdbl = sq_tail;
    cid = cmd_id;
    cmd_id = (cmd_id + 1) & NVME_ENTRY_CID_MASK;
}
__host__ __device__ void AdminQueuePair::fill_sq_with_ns(uint32_t cid, uint32_t pos, uint32_t opcode, uint32_t nsid, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12, uint32_t req_id)
{
    sq[pos * 16 + 0] = opcode | (cid << 16);
    sq[pos * 16 + 1] = nsid;
    sq[pos * 16 + 6] = prp1 & 0xffffffff;
    sq[pos * 16 + 7] = prp1 >> 32;
    sq[pos * 16 + 8] = prp2 & 0xffffffff;
    sq[pos * 16 + 9] = prp2 >> 32;
    sq[pos * 16 + 10] = dw10;
    sq[pos * 16 + 11] = dw11;
    sq[pos * 16 + 12] = dw12;
    if (cmd_id_to_req_id)
        cmd_id_to_req_id[cid % queue_depth] = req_id;
    if (cmd_id_to_sq_pos)
        cmd_id_to_sq_pos[cid % queue_depth] = pos;
    if (sq_entry_busy)
        sq_entry_busy[pos] = true;
}

__host__ uint32_t AdminQueuePair::set_num_queues(uint16_t nsqr, uint16_t ncqr)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_SET_FEATURES, 0x0, 0x0, NVME_FEATURE_ID_NUM_QUEUES, 
        ((ncqr-1) << 16) | (nsqr-1) 
    );
    uint32_t status;
    poll(status, cid);
    return status;
}

__host__ uint32_t AdminQueuePair::get_num_queues(uint16_t &nsqa, uint16_t &ncqa)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_GET_FEATURES, 0x0, 0x0, NVME_FEATURE_ID_NUM_QUEUES, 0x0
    );
    uint32_t dw0;
    uint32_t status;
    poll_with_dw0(status, cid, dw0);
    nsqa = (dw0 & 0xffff) + 1;
    ncqa = ((dw0 >> 16) & 0xffff) + 1;
    return status;
}

__host__ uint32_t AdminQueuePair::identify(uint8_t cns, uint16_t cntid, uint32_t nsid, uint64_t prp1)
{
    uint32_t cid;
    submit_with_ns(
        cid, NVME_ADMIN_OPCODE_IDENTIFY, nsid, prp1, 0x0, cns | (cntid << 16), 0x0
    );
    uint32_t status;
    poll(status, cid);
    return status;
}

__host__ uint32_t AdminQueuePair::create_cq_cont(uint16_t cqid, uint64_t cq_phys, uint16_t queue_depth)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_CREATE_CQ, cq_phys, 0x0, cqid | ((queue_depth-1) << 16), 0x1
    );
    uint32_t status;
    poll(status, cid);
    return status;
}

__host__ uint32_t AdminQueuePair::create_sq_cont(uint16_t sqid, uint16_t cqid, uint64_t sq_phys, uint16_t queue_depth)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_CREATE_SQ, sq_phys, 0x0, sqid | ((queue_depth-1) << 16), (cqid << 16) | 0x1
    );
    uint32_t status;
    poll(status, cid);
    return status;
}

__host__ uint32_t AdminQueuePair::delete_sq(uint16_t sqid)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_DELETE_SQ, 0x0, 0x0, sqid, 0x0
    );
    uint32_t status;
    poll(status, cid);
    return status;
}

__host__ uint32_t AdminQueuePair::delete_cq(uint16_t cqid)
{
    uint32_t cid;
    submit(
        cid, NVME_ADMIN_OPCODE_DELETE_SQ, 0x0, 0x0, cqid, 0x0
    );
    uint32_t status;
    poll(status, cid);
    return status;
}