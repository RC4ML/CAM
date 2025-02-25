#ifndef __AEOLUS_UTIL_CUH
#define __AEOLUS_UTIL_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include "cufile.h"

// NVMe BAR0 register sizes and offsets.

#define NVME_BAR0_SIZE      0x4000
#define NVME_REG_CC         0x14    // addr: controller configuration
#define NVME_REG_CC_EN      0x1     // mask: enable controller
#define NVME_REG_CSTS       0x1c    // addr: controller status
#define NVME_REG_CSTS_RDY   0x1     // mask: controller ready
#define NVME_REG_AQA        0x24    // addr: admin queue attributes
#define NVME_REG_ASQ        0x28    // addr: admin submission queue base addr
#define NVME_REG_ACQ        0x30    // addr: admin completion queue base addr
#define NVME_REG_SQTDBL     0x1000  // addr: submission queue 0 tail doorbell
#define NVME_REG_CQHDBL     0x1004  // addr: completion queue 0 sq_tail doorbell

// NVMe admin opcode
#define NVME_ADMIN_OPCODE_DELETE_SQ     0x00
#define NVME_ADMIN_OPCODE_CREATE_SQ     0x01
#define NVME_ADMIN_OPCODE_DELETE_CQ     0x04
#define NVME_ADMIN_OPCODE_CREATE_CQ     0x05
#define NVME_ADMIN_OPCODE_IDENTIFY      0x06
#define NVME_ADMIN_OPCODE_SET_FEATURES  0x09
#define NVME_ADMIN_OPCODE_GET_FEATURES  0x0a

// NVMe opcode
#define NVME_OPCODE_READ                0x02
#define NVME_OPCODE_WRITE               0x01

// NVMe feature ID.
#define NVME_FEATURE_ID_NUM_QUEUES      0x07

// NVMe field masks.
#define NVME_ENTRY_PHASE_MASK   0x10000
#define NVME_ENTRY_CID_MASK     0xffff  // mask: command id
#define NVME_ENTRY_SC_MASK      0xff    // mask: status code
#define NVME_ENTRY_SQ_HEAD_MASK 0xffff
#define NVME_RW_LIMITED_RETRY_MASK 0x80000000

// NVMe misc
#define NVME_BROADCAST_NSID     0xffffffff
#define NVME_SQ_ENTRY_SIZE      64
#define NVME_CQ_ENTRY_SIZE      16
#define NVME_DBLSTRIDE          8

// Other constants.

#define AEOLUS_HOST_PGSIZE        4096
#define AEOLUS_DEVICE_PGSIZE      0x10000
#define AEOLUS_ADMIN_QUEUE_DEPTH  64
#define AEOLUS_WARP_SIZE          32
#define AEOLUS_LB_SIZE            512
#define AEOLUS_NUM_THREADS_PER_BLOCK 512
#define AEOLUS_MAX_NUM_REQUESTS   4000000

#define AEOLUS_MAX_NUM_QUEUES     -1
#define AEOLUS_MAX_DATA_TRANSFER  -1

// Check cuda errors.

#define AEOLUS_CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(1);
    }
}

inline bool isPowerOfTwo(int num) {
    return (num > 0) && ((num & (num - 1)) == 0);
}

inline uint64_t longrand(uint64_t max, uint64_t min = 0) {
    return min + (((unsigned long)rand() << 31 | rand()) % (max - min));
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((b) < (a) ? (a) : (b))
#define CEIL(a, b) (((a)+(b)-1) / (b))

//
// cuda driver error description
//
static inline const char *GetCuErrorString(CUresult curesult) {
	const char *descp;
	if (cuGetErrorName(curesult, &descp) != CUDA_SUCCESS)
		descp = "unknown cuda error";
	return descp;
}

//
// cuFile APIs return both cuFile specific error codes as well as POSIX error codes
// for ease, the below template can be used for getting the error description depending
// on its type.

// POSIX
template<class T,
	typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	status = std::abs(status);
	return IS_CUFILE_ERR(status) ?
		std::string(CUFILE_ERRSTR(status)) : std::string(strerror(status));
}

// CUfileError_t
template<class T,
	typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
	if (IS_CUDA_ERR(status))
		errStr.append(".").append(GetCuErrorString(status.cu_err));
	return errStr;
}

#define AEOLUS_CUFILE_CHECK(ans) cufileAssert((ans), __FILE__, __LINE__)

inline void cufileAssert(CUfileError_t status, const char *file, int line, bool abort = true)
{
    if (status.err != CU_FILE_SUCCESS)
    {
        fprintf(stderr, "CUfileAssert: %s %s %d\n", cuFileGetErrorString(status).c_str(), file, line);
        if (abort)
            exit(1);
    }
}

#endif