/*   SPDX-License-Identifier: BSD-3-Clause
 *   Copyright (C) 2016 Intel Corporation.
 *   All rights reserved.
 */


#ifndef GPUSSD_BASELINE_H
#define GPUSSD_BASELINE_H


#include <cuda.h>
// #include "cuda_runtime.h"
#include <cuda_runtime.h>
#include <iostream>
#include <QDMAController.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <thread>
#include "threadPool.h"

#include "spdk/stdinc.h"

#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/nvme.h"
#include "spdk/nvme_zns.h"
#include "spdk/string.h"
#include "spdk/vmd.h"

#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__

#define MAX_EMBED_NUM  1000000
struct ctrlr_entry {
    struct spdk_nvme_ctrlr* ctrlr;
    char name[1024];
};

struct ns_entry {
    int32_t id;
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
    struct spdk_nvme_qpair* qpair;
};





static  void read_complete(void* arg, const struct spdk_nvme_cpl* completion) ;
//static   int thread_runner(int32_t dev_index);
static    int thread_runner2(int32_t dev_index) ;
static    bool probe_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid, struct spdk_nvme_ctrlr_opts* opts);
static    void attach_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid, struct spdk_nvme_ctrlr* ctrlr, const struct spdk_nvme_ctrlr_opts* opts);
static void register_ns(struct spdk_nvme_ctrlr* ctrlr, struct spdk_nvme_ns* ns) ;
void task_submit(int64_t embed_num, u_int64_t embed_id,uintptr_t *dev_addr);
// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr);    
int rc4ml_spdk_init(u_int32_t emb_width);
 static void alloc_qpair() ;
inline std::pair<int64_t, int64_t> getEmbedAddr(int32_t embed_id) ;
void spdkmap(void * map_ptr,size_t  pool_size,uint64_t  phy_addr);
void clear_wait_flag();

//* new function

static  void write_complete(void* arg, const struct spdk_nvme_cpl* completion);
static int thread_runner3(int32_t dev_index);
void task_submit_write(int64_t embed_num, u_int64_t embed_id,uintptr_t *dev_addr);
void clear_wait_flag_write();
void cam_init(u_int32_t emb_width);
void* alloc_gpu(int64_t size);
void free_gpu(void* p);
void cam_clean_up(void);

void seq_read_submit(u_int64_t start_lba, u_int64_t num_blocks,uintptr_t dev_addr);
void seq_write_submit(u_int64_t start_lba, u_int64_t num_blocks,uintptr_t dev_addr);


void cam_gemm_read(u_int64_t * lba_array, u_int64_t req_num,uintptr_t dev_addr);
void cam_gemm_write(u_int64_t * lba_array, u_int64_t req_num,uintptr_t dev_addr);

#endif