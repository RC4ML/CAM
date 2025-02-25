#ifndef __AEOLUS_QUEUE_CUH
#define __AEOLUS_QUEUE_CUH

#include <stdint.h>
#include <assert.h>
#include "util.cuh"
#include "log.cuh"

/**
 * @brief Abstraction of an SSD SQ-CQ pair.
 * 
 */
class QueuePair
{
public:
    uint32_t *cmd_id_to_req_id;
    uint32_t *cmd_id_to_sq_pos;
    bool     *sq_entry_busy;
    uint32_t sq_tail;
    uint32_t cq_head;
    uint32_t cmd_id; // also number of commands submitted
    uint32_t *sqtdbl, *cqhdbl;
    uint32_t num_completed;
    volatile uint32_t *sq;
    volatile uint32_t *cq;
protected:
    uint32_t namespace_id;
    uint32_t queue_depth;

public:
    inline __host__ __device__ QueuePair()
    {
    }

    inline __host__ __device__ QueuePair(volatile uint32_t *sq, volatile uint32_t *cq, uint32_t namespace_id, uint32_t *sqtdbl, uint32_t *cqhdbl, uint32_t queue_depth, uint32_t *cmd_id_to_req_id = nullptr, uint32_t *cmd_id_to_sq_pos = nullptr, bool *sq_entry_busy = nullptr)
        : sq(sq), cq(cq), sq_tail(0), cq_head(0), cmd_id(0), namespace_id(namespace_id), sqtdbl(sqtdbl), cqhdbl(cqhdbl), cmd_id_to_req_id(cmd_id_to_req_id), cmd_id_to_sq_pos(cmd_id_to_sq_pos), sq_entry_busy(sq_entry_busy), queue_depth(queue_depth), num_completed(0)
    {
    }

    __host__ __device__ void submit(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0);

    __device__ void submit_fence(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0);

    __host__ __device__ void fill_sq(uint32_t cid, uint32_t pos, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0, uint32_t req_id = 0xffffffff);

    __host__ __device__ void poll(uint32_t &code, uint32_t cid);

    __host__ __device__ void poll_with_dw0(uint32_t &code, uint32_t cid, uint32_t &dw0);
};

/**
 * @brief Abstraction of an SSD IO queue pair.
 * 
 */
class IoQueuePair : public QueuePair {
public:
    inline __host__ IoQueuePair(
        volatile uint32_t *sq, 
        volatile uint32_t *cq, 
        uint32_t namespace_id, 
        uint32_t *sqtdbl, 
        uint32_t *cqhdbl, 
        uint32_t queue_depth, 
        uint32_t *cmd_id_to_req_id = nullptr, 
        uint32_t *cmd_id_to_sq_pos = nullptr, 
        bool *sq_entry_busy = nullptr
    ) : QueuePair(
        sq, cq, namespace_id, sqtdbl, cqhdbl, queue_depth, 
        cmd_id_to_req_id, cmd_id_to_sq_pos, sq_entry_busy
    )
    {
        // AEOLUS_LOG_INFO("IoQueuePair sqtdbl %p cqhdbl %p", sqtdbl, cqhdbl);
    }

    __device__ void poll_range(uint32_t &code, int expected_sq_head, bool should_break);
    __device__ void poll_multiple(uint32_t &code, int cnt);
    __device__ void poll_until_sq_entry_free(uint32_t &code, int expected_sq_pos);
};

/**
 * @brief Abstraction of an SSD admin queue pair.
 * 
 */
class AdminQueuePair : public QueuePair {
public:
    inline __host__ AdminQueuePair(
        volatile uint32_t *sq, 
        volatile uint32_t *cq, 
        uint32_t namespace_id, 
        uint32_t *sqtdbl, 
        uint32_t *cqhdbl, 
        uint32_t queue_depth, 
        uint32_t *cmd_id_to_req_id = nullptr, 
        uint32_t *cmd_id_to_sq_pos = nullptr, 
        bool *sq_entry_busy = nullptr
    ) : QueuePair(
        sq, cq, namespace_id, sqtdbl, cqhdbl, queue_depth, 
        cmd_id_to_req_id, cmd_id_to_sq_pos, sq_entry_busy
    )
    {
        // AEOLUS_LOG_INFO("AdminQueuePair sqtdbl %p cqhdbl %p", sqtdbl, cqhdbl);
    }
    __host__ __device__ void submit_with_ns(uint32_t &cid, uint32_t opcode, uint32_t nsid, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0);
    __host__ __device__ void fill_sq_with_ns(uint32_t cid, uint32_t pos, uint32_t opcode, uint32_t nsid, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0, uint32_t req_id = 0xffffffff);
    __host__ uint32_t set_num_queues(uint16_t nsqr, uint16_t ncqr);
    __host__ uint32_t get_num_queues(uint16_t &nsqa, uint16_t &ncqa);
    __host__ uint32_t identify(uint8_t cns, uint16_t cntid, uint32_t nsid, uint64_t prp1);
    __host__ uint32_t create_cq_cont(uint16_t cqid, uint64_t cq_phys, uint16_t queue_depth);
    __host__ uint32_t create_sq_cont(uint16_t sqid, uint16_t cqid, uint64_t sq_phys, uint16_t queue_depth);
    __host__ uint32_t delete_sq(uint16_t sqid);
    __host__ uint32_t delete_cq(uint16_t cqid);
};

#endif