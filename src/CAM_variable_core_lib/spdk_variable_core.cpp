

#include "spdk_variable_core.h"


#include "GPU_memory_management.hpp"
#include <iostream>
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

static const int32_t max_dev_num = 12;

static std::vector<ctrlr_entry*> g_controllers;
static std::vector<ns_entry*> g_namespaces;
static struct spdk_nvme_transport_id g_trid = {};

static int64_t total_pending = 0;
static std::vector<int64_t> pending_io_per_dev;

static GPUMemCtl* gpuMemCtl;

static const int64_t lba_size = 512;
static int64_t embed_entry_width ;
static int64_t embed_entry_lba;
static int64_t total_core_num ;

static int64_t element_per_buffer[max_dev_num];
static std::pair<int64_t, int64_t> *thread_buffer[max_dev_num];

ThreadPool threadPool(max_dev_num);
//const bool enable_multi_thread = true;

std::vector<std::future<int>> wait_flag;

// const int64_t embed_num = 600000;
// int32_t embed_id[embed_num];
// uintptr_t dev_addr[embed_num];
const size_t pool_size = 70UL * 1024 * 1024 * 1024;

//* new arguments
static int64_t element_per_buffer_write[max_dev_num];
std::vector<std::future<int>> wait_flag_write;
ThreadPool threadPool_write(max_dev_num);
static std::pair<int64_t, int64_t> *thread_buffer_write[max_dev_num];  
static std::vector<int64_t> pending_io_per_dev_write;

static  void
read_complete(void* arg, const struct spdk_nvme_cpl* completion) {
	struct ns_entry* ns_entry = (struct ns_entry*)arg;

    /* See if an error occurred. If so, display information
     * about it, and set completion value so that I/O
     * caller is aware that an error occurred.
     */
    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(ns_entry->qpair, (struct spdk_nvme_cpl*)completion);
        fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
        fprintf(stderr, "Read I/O failed, aborting run\n");
        exit(1);
    }
    int64_t total_pending = 0;
    --total_pending;
    --pending_io_per_dev[ns_entry->id];
}

int rc4ml_gdr_init() {
    std::cout<<"13:26"<<std::endl;
    gpuMemCtl = GPUMemCtl::getInstance(0, pool_size);

    if (gpuMemCtl->chechPhyContiguous() == false) {
        printf("GPU memory PhyAddr is not contiguous\n");
        return 1;
    }

    void* dev_ptr = gpuMemCtl->getDevPtr();
    void* map_ptr = gpuMemCtl->getMapDevPtr();
    auto phy_addr = gpuMemCtl->mapV2P(dev_ptr);

    if (rc4ml_mem_register(map_ptr, pool_size, phy_addr) != 0) {
        fprintf(stderr, "rc4ml_mem_register() failed\n");
        return 1;
    }

    return 0;
}

static inline void* devPtr2Map(void* dev_ptr) {
    void* dev_ptr_base = gpuMemCtl->getDevPtr();
    void* map_ptr_base = gpuMemCtl->getMapDevPtr();

    return (void*)((uintptr_t)map_ptr_base + ((uintptr_t)dev_ptr - (uintptr_t)dev_ptr_base));
}

// <dev_id, lba_addr>
inline std::pair<int64_t, int64_t> getEmbedAddr(int32_t embed_id) {
    //printf("embed_entry_lba : %d \n",embed_entry_lba);
    //printf("g_namespaces.size() : %d \n",g_namespaces.size());
    int64_t dev_id = embed_id % g_namespaces.size();
    int64_t lba_addr = embed_id / g_namespaces.size() * embed_entry_lba;
    
    return std::make_pair(dev_id, lba_addr);
}

static void
register_ns(struct spdk_nvme_ctrlr* ctrlr, struct spdk_nvme_ns* ns) {
    struct ns_entry* entry;

    if (!spdk_nvme_ns_is_active(ns)) {
        return;
    }

    entry = (struct ns_entry*)malloc(sizeof(struct ns_entry));
    if (entry == NULL) {
        perror("ns_entry malloc");
        exit(1);
    }

    entry->ctrlr = ctrlr;
    entry->ns = ns;
    entry->id = g_namespaces.size();
    g_namespaces.push_back(entry);
    pending_io_per_dev.push_back(0);
    pending_io_per_dev_write.push_back(0);
    printf("  Namespace ID: %d size: %juGB\n", spdk_nvme_ns_get_id(ns),
           spdk_nvme_ns_get_size(ns) / 1000000000);
}






static int thread_runner2(int32_t dev_index) {
    bool submit_end = false;
    int64_t local_buffer_index = 0;
    struct ns_entry* ns_entry = g_namespaces[dev_index];

    while(!submit_end || pending_io_per_dev[dev_index] > 0) {
        if(!submit_end) {
            while(local_buffer_index < element_per_buffer[dev_index]) {
                if(thread_buffer[dev_index][local_buffer_index].first == -1) {
                    submit_end = true;
                    break;
                }
                auto lba_addr = thread_buffer[dev_index][local_buffer_index].first;

                void* map_addr = (void*)thread_buffer[dev_index][local_buffer_index].second;

                ++pending_io_per_dev[dev_index];

                auto rc = spdk_nvme_ns_cmd_read(ns_entry->ns, ns_entry->qpair, map_addr,
                                                lba_addr, /* LBA start */
                                                embed_entry_lba, /* number of LBAs */
                                                 read_complete, (void*)ns_entry, 0);
                if(rc != 0) {
                    fprintf(stderr, "Starting read I/O failed at dev %d index %ld\n", dev_index, local_buffer_index);
                    fprintf(stderr, "lbaddr : %ld\n",lba_addr);
                    fprintf(stderr, "map_addr : %p\n",map_addr);
                    fprintf(stderr, "embed_entry_lba : %ld\n",embed_entry_lba);
                    fprintf(stderr, "embed_entry_width : %ld\n",embed_entry_width);
                    exit(1);
                }

                ++local_buffer_index;
            }
        }

        spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
    }

    return 0;
}

static int thread_runner_variablecore(int32_t thread_index){
    bool submit_all_end = false;
    bool all_complete = false;
    std::vector<int64_t> thread_dev_index;
    std::vector<struct ns_entry*> ns_entry_array;
    std::vector<int64_t> local_buffer_index;
    for(uint64_t i =thread_index;i<g_namespaces.size();i+=total_core_num){
        thread_dev_index.push_back(i);
        ns_entry_array.push_back(g_namespaces[i]);
        local_buffer_index.push_back(0);
    }
    
    u_int32_t dev_num = thread_dev_index.size();
    //printf("dev_num : %d\n",dev_num);
    while(!submit_all_end || !all_complete){
            //if(thread_index ==2)
              // printf("submit_all_end : %d all_complete : %d\n",submit_all_end,all_complete);
        if(!submit_all_end){
            submit_all_end = true;
            for(int i = 0; i < dev_num; i++){
                if(local_buffer_index[i] < element_per_buffer[thread_dev_index[i]] && thread_buffer[thread_dev_index[i]][local_buffer_index[i]].first != -1){
                    
                    auto lba_addr = thread_buffer[thread_dev_index[i]][local_buffer_index[i]].first;
                    void* map_addr = (void*)thread_buffer[thread_dev_index[i]][local_buffer_index[i]].second;
                    ++pending_io_per_dev[thread_dev_index[i]];
                    auto rc = spdk_nvme_ns_cmd_read(ns_entry_array[i]->ns, ns_entry_array[i]->qpair, map_addr,
                                                lba_addr, /* LBA start */
                                                embed_entry_lba, /* number of LBAs */
                                                read_complete, (void*)ns_entry_array[i], 0);
                    if(rc != 0) {
                        fprintf(stderr, "Starting read I/O failed at dev %d index %ld\n", thread_dev_index[i], local_buffer_index[i]);
                        fprintf(stderr, "lbaddr : %ld\n",lba_addr);
                        fprintf(stderr, "map_addr : %p\n",map_addr);
                        fprintf(stderr, "embed_entry_lba : %ld\n",embed_entry_lba);
                        fprintf(stderr, "embed_entry_width : %ld\n",embed_entry_width);
                        exit(1);
                    }
                    ++local_buffer_index[i];
                }
            }
            for(int i = 0; i < dev_num; i++){
                if(thread_buffer[thread_dev_index[i]][local_buffer_index[i]].first != -1){
                    submit_all_end = false;
                    break;
                }
            }
        }
        all_complete = true;
        for(int i = 0; i < dev_num; i++){
            if(pending_io_per_dev[thread_dev_index[i]] > 0){
                all_complete = false;
                spdk_nvme_qpair_process_completions(ns_entry_array[i]->qpair, 0);
            }
        }
    }
    return 0;
}

// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr) {
void task_submit(int64_t embed_num, u_int64_t embed_id,uintptr_t *dev_addr) {
    //printf("entering task submit!\n");
    
    for(int32_t i=0; i<(int32_t)total_core_num; ++i) {
        
        wait_flag.emplace_back(threadPool.enqueue(thread_runner2, i));
    }
    for(int32_t i=0; i< g_namespaces.size(); ++i) {
        element_per_buffer[i] = 0;
    }
    int32_t * p_embed_id = (int32_t *)embed_id;
    for(int64_t i = 0; i < embed_num; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        auto [dev_id, lba_addr] = getEmbedAddr(p_embed_id[i]);
        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        //void* map_addr = dev_addr+embed_entry_width*i ;
        //map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        void* map_addr = devPtr2Map((void*)dev_addr[i]);
        //printf("map_addr : %p\n",map_addr);
       //printf("1111\n",i);
        thread_buffer[dev_id][element_per_buffer[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer[i][element_per_buffer[i]] = std::make_pair(-1, -1);
        ++element_per_buffer[i];
    }

    
}

static void alloc_qpair() {
    for (auto ns_entry : g_namespaces) {
        /*
         * Allocate an I/O qpair that we can use to submit read/write requests
         *  to namespaces on the controller.  NVMe controllers typically support
         *  many qpairs per controller.  Any I/O qpair allocated for a controller
         *  can submit I/O to any namespace on that controller.
         *
         * The SPDK NVMe driver provides no synchronization for qpair accesses -
         *  the application must ensure only a single thread submits I/O to a
         *  qpair, and that same thread must also check for completions on that
         *  qpair.  This enables extremely efficient I/O processing by making all
         *  I/O operations completely lockless.
         */
        spdk_nvme_io_qpair_opts opts = {};
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ns_entry->ctrlr, &opts, sizeof(spdk_nvme_io_qpair_opts));
        opts.io_queue_requests = 1048576;
        // printf("opts.sq.buffer_size: %ld\n", opts.sq.buffer_size);
        // printf("opts.cq.buffer_size: %ld\n", opts.cq.buffer_size);
        ns_entry->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ns_entry->ctrlr, &opts, sizeof(spdk_nvme_io_qpair_opts));
        if (ns_entry->qpair == NULL) {
            printf("ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed\n");
            return;
        }
    }
}

void release_qpair() {
    /*
     * Free the I/O qpair.  This typically is done when an application exits.
     *  But SPDK does support freeing and then reallocating qpairs during
     *  operation.  It is the responsibility of the caller to ensure all
     *  pending I/O are completed before trying to free the qpair.
     */
    for (auto ns_entry : g_namespaces) {
        spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
    }
}

static bool
probe_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid, struct spdk_nvme_ctrlr_opts* opts) {
    printf("Attaching to %s\n", trid->traddr);

    return true;
}

static void
attach_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid, struct spdk_nvme_ctrlr* ctrlr, const struct spdk_nvme_ctrlr_opts* opts) {
    int nsid;
    printf("attach_cb \n");
    struct ctrlr_entry* entry;
    struct spdk_nvme_ns* ns;
    const struct spdk_nvme_ctrlr_data* cdata;

    entry = (struct ctrlr_entry*)malloc(sizeof(struct ctrlr_entry));
    if (entry == NULL) {
        perror("ctrlr_entry malloc");
        exit(1);
    }

    printf("Attached to %s\n", trid->traddr);

    /*
     * spdk_nvme_ctrlr is the logical abstraction in SPDK for an NVMe
     *  controller.  During initialization, the IDENTIFY data for the
     *  controller is read using an NVMe admin command, and that data
     *  can be retrieved using spdk_nvme_ctrlr_get_data() to get
     *  detailed information on the controller.  Refer to the NVMe
     *  specification for more details on IDENTIFY for NVMe controllers.
     */
    cdata = spdk_nvme_ctrlr_get_data(ctrlr);

    snprintf(entry->name, sizeof(entry->name), "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);

    entry->ctrlr = ctrlr;
    g_controllers.push_back(entry);

    /*
     * Each controller has one or more namespaces.  An NVMe namespace is basically
     *  equivalent to a SCSI LUN.  The controller's IDENTIFY data tells us how
     *  many namespaces exist on the controller.  For Intel(R) P3X00 controllers,
     *  it will just be one namespace.
     *
     * Note that in NVMe, namespace IDs start at 1, not 0.
     */
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == NULL) {
            continue;
        }
        register_ns(ctrlr, ns);
    }
}

void
rc4ml_spdk_cleanup(void) {
    struct spdk_nvme_detach_ctx* detach_ctx = NULL;

    for(int i = 0; i < max_dev_num; i++) {
        free(thread_buffer[i]);
    }

    for(int i = 0; i < max_dev_num; i++) {
        free(thread_buffer_write[i]);
    }

    release_qpair();

    for (auto ns_entry : g_namespaces) {
        free(ns_entry);
    }
    g_namespaces.clear();

    for (auto ctrlr_entry : g_controllers) {
        spdk_nvme_detach_async(ctrlr_entry->ctrlr, &detach_ctx);
        free(ctrlr_entry);
    }
    g_controllers.clear();

    if (detach_ctx) {
        spdk_nvme_detach_poll(detach_ctx);
    }

    spdk_env_fini();
}


int rc4ml_spdk_init(u_int32_t emb_width) {
    struct spdk_env_opts opts;

    /*
     * SPDK relies on an abstraction around the local environment
     * named env that handles memory allocation and PCI device operations.
     * This library must be initialized first.
     *
     */

    embed_entry_width = emb_width;
    embed_entry_lba = embed_entry_width / lba_size;
    spdk_nvme_trid_populate_transport(&g_trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid.subnqn, sizeof(g_trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);

    spdk_env_opts_init(&opts);

    opts.name = "gpussd_baseline";
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "Unable to initialize SPDK env\n");
        return 1;
    }

    printf("Initializing NVMe Controllers\n");

    /*
     * Start the SPDK NVMe enumeration process.  probe_cb will be called
     *  for each NVMe controller found, giving our application a choice on
     *  whether to attach to each controller.  attach_cb will then be
     *  called for each controller after the SPDK NVMe driver has completed
     *  initializing the controller we chose to attach.
     */

    if (spdk_nvme_probe(&g_trid, NULL, probe_cb, attach_cb, NULL) != 0) {
        fprintf(stderr, "spdk_nvme_probe() failed\n");
        return 1;
    }

    if (g_controllers.empty()) {
        fprintf(stderr, "no NVMe controllers found\n");
        return 1;
    }

    printf("Initialization complete.\n");
    alloc_qpair();
    for(int i = 0; i < max_dev_num; i++) {
        thread_buffer[i] = (std::pair<int64_t, int64_t>*)malloc(sizeof(std::pair<int64_t, int64_t>) * 1048576);
    }
    for(int i = 0; i < max_dev_num; i++) {
        thread_buffer_write[i] = (std::pair<int64_t, int64_t>*)malloc(sizeof(std::pair<int64_t, int64_t>) * 1048576);
    }
    return 0;
}



void spdkmap(void * map_ptr,size_t  pool_size,uint64_t  phy_addr)
{
    if (rc4ml_mem_register(map_ptr, pool_size, phy_addr) != 0) {
        fprintf(stderr, "rc4ml_mem_register() failed\n");
    }else{
        printf("**********************rc4ml_mem_register success*********\n");
    }
}

void clear_wait_flag()
{
    // for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
    //     wait_flag[i].get();
    // }
    for(uint32_t i = 0; i < total_core_num; i++){
        //printf("wait flag : %d\n",i);
        wait_flag[i].get();
    }
    wait_flag.clear();
}



//* ===========================new function======================================  *//

static  void
write_complete(void* arg, const struct spdk_nvme_cpl* completion) {
	struct ns_entry* ns_entry = (struct ns_entry*)arg;

    /* See if an error occurred. If so, display information
     * about it, and set completion value so that I/O
     * caller is aware that an error occurred.
     */
    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(ns_entry->qpair, (struct spdk_nvme_cpl*)completion);
        fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
        fprintf(stderr, "Write I/O failed, aborting run\n");
        exit(1);
    }
    int64_t total_pending = 0;
    --total_pending;
    --pending_io_per_dev_write[ns_entry->id];
}



static int thread_runner3(int32_t dev_index) {
    bool submit_end = false;
    int64_t local_buffer_index = 0;
    struct ns_entry* ns_entry = g_namespaces[dev_index];

    while(!submit_end || pending_io_per_dev_write[dev_index] > 0) {
        if(!submit_end) {
            while(local_buffer_index < element_per_buffer_write[dev_index]) {
                if(thread_buffer_write[dev_index][local_buffer_index].first == -1) {
                    submit_end = true;
                    break;
                }
                auto lba_addr = thread_buffer_write[dev_index][local_buffer_index].first;

                void* map_addr = (void*)thread_buffer_write[dev_index][local_buffer_index].second;

                ++pending_io_per_dev_write[dev_index];

                auto rc = spdk_nvme_ns_cmd_write(ns_entry->ns, ns_entry->qpair, map_addr,
                                                lba_addr, /* LBA start */
                                                embed_entry_lba, /* number of LBAs */
                                                write_complete, (void*)ns_entry, 0);
                if(rc != 0) {
                   
                    fprintf(stderr, "Starting write I/O failed at dev %d index %ld\n", dev_index, local_buffer_index);
                    fprintf(stderr, "lbaddr : %ld\n",lba_addr);
                    fprintf(stderr, "map_addr : %p\n",map_addr);
                    fprintf(stderr, "embed_entry_lba : %ld\n",embed_entry_lba);
                    fprintf(stderr, "embed_entry_width : %ld\n",embed_entry_width);
                    exit(1);
                }

                ++local_buffer_index;
            }
        }

        spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
    }

    return 0;
}


static int thread_runner_variablecore_write(int32_t thread_index){
    bool submit_all_end = false;
    bool all_complete = false;
    std::vector<int64_t> thread_dev_index;
    std::vector<struct ns_entry*> ns_entry_array;
    std::vector<int64_t> local_buffer_index;
    for(uint64_t i =thread_index;i<g_namespaces.size();i+=total_core_num){
        thread_dev_index.push_back(i);
        ns_entry_array.push_back(g_namespaces[i]);
        local_buffer_index.push_back(0);
    }
    
    u_int32_t dev_num = thread_dev_index.size();
    //printf("dev_num : %d\n",dev_num);
    while(!submit_all_end || !all_complete){
            //if(thread_index ==2)
              // printf("submit_all_end : %d all_complete : %d\n",submit_all_end,all_complete);
        if(!submit_all_end){
            submit_all_end = true;
            for(int i = 0; i < dev_num; i++){
                if(local_buffer_index[i] < element_per_buffer_write[thread_dev_index[i]] && thread_buffer_write[thread_dev_index[i]][local_buffer_index[i]].first != -1){
                    
                    auto lba_addr = thread_buffer_write[thread_dev_index[i]][local_buffer_index[i]].first;
                    void* map_addr = (void*)thread_buffer_write[thread_dev_index[i]][local_buffer_index[i]].second;
                    ++pending_io_per_dev_write[thread_dev_index[i]];
                    auto rc = spdk_nvme_ns_cmd_write(ns_entry_array[i]->ns, ns_entry_array[i]->qpair, map_addr,
                                                lba_addr, /* LBA start */
                                                embed_entry_lba, /* number of LBAs */
                                                write_complete, (void*)ns_entry_array[i], 0);
                    if(rc != 0) {
                        fprintf(stderr, "Starting write I/O failed at dev %d index %ld\n", thread_dev_index[i], local_buffer_index[i]);
                        fprintf(stderr, "lbaddr : %ld\n",lba_addr);
                        fprintf(stderr, "map_addr : %p\n",map_addr);
                        fprintf(stderr, "embed_entry_lba : %ld\n",embed_entry_lba);
                        fprintf(stderr, "embed_entry_width : %ld\n",embed_entry_width);
                        exit(1);
                    }
                    ++local_buffer_index[i];
                }
            }
            for(int i = 0; i < dev_num; i++){
                if(thread_buffer_write[thread_dev_index[i]][local_buffer_index[i]].first != -1){
                    submit_all_end = false;
                    break;
                }
            }
        }
        all_complete = true;
        for(int i = 0; i < dev_num; i++){
            if(pending_io_per_dev_write[thread_dev_index[i]] > 0){
                all_complete = false;
                spdk_nvme_qpair_process_completions(ns_entry_array[i]->qpair, 0);
            }
        }
    }
    
        return 0;
}


// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr) {
void task_submit_write(int64_t embed_num, u_int64_t embed_id,uintptr_t *dev_addr) {
    //printf("entering task submit!\n");
    
    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        element_per_buffer_write[i] = 0;
        wait_flag_write.emplace_back(threadPool_write.enqueue(thread_runner3, i));
    }

    int32_t * p_embed_id = (int32_t *)embed_id;
    for(int64_t i = 0; i < embed_num; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        auto [dev_id, lba_addr] = getEmbedAddr(p_embed_id[i]);
        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        void* map_addr = devPtr2Map((void*)dev_addr[i]);
        // void* map_addr = dev_addr+embed_entry_width*i ;
        // map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        //printf("map_addr : %p\n",map_addr);
       //printf("1111\n",i);
        thread_buffer_write[dev_id][element_per_buffer_write[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer_write[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer_write[i][element_per_buffer_write[i]] = std::make_pair(-1, -1);
        ++element_per_buffer_write[i];
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        wait_flag_write[i].get();
    }
}

void clear_wait_flag_write()
{
    for(int32_t i=0; i<total_core_num; ++i) {
        wait_flag_write[i].get();
    }
    wait_flag_write.clear();
}



void cam_init(u_int32_t emb_width, u_int32_t core_num)
{
    int rc;
    total_core_num = core_num;
    rc = rc4ml_spdk_init(emb_width);
    if (rc != 0) {
        fprintf(stderr, "rc4ml_spdk_init() failed\n");
        rc = 1;
    }
    rc = rc4ml_gdr_init();
    if (rc != 0) {
        fprintf(stderr, "rc4ml_gdr_init() failed\n");
        rc = 1;
    }
    std::cout<<"initialization done."<<std::endl;
}

void* alloc_gpu(int64_t size)
{
    return gpuMemCtl->alloc(size);
}

void free_gpu(void* p)
{
    gpuMemCtl->free(p);
}

void cam_clean_up(void)
{
    rc4ml_spdk_cleanup();
    GPUMemCtl::cleanCtx();
}


// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr) {
void seq_read_submit(u_int64_t start_lba, u_int64_t num_blocks,uintptr_t dev_addr) {
    //printf("entering task submit!\n");
    
    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        element_per_buffer[i] = 0;
        wait_flag.emplace_back(threadPool.enqueue(thread_runner2, i));
    }

    //int32_t * p_embed_id = (int32_t *)embed_id;
    for(u_int64_t i = 0; i < num_blocks; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        int64_t dev_id = (start_lba+i) % g_namespaces.size();
        int64_t lba_addr = (start_lba+i)/ g_namespaces.size()*8 ;//512*8 = 4096 

        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        //void* map_addr = dev_addr+embed_entry_width*i ;
        //map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        void* map_addr = devPtr2Map((void*)(dev_addr+i*4UL*1024));
        //printf("map_addr : %p\n",map_addr);UL
       //printf("1111\n",i);
        thread_buffer[dev_id][element_per_buffer[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer[i][element_per_buffer[i]] = std::make_pair(-1, -1);
        ++element_per_buffer[i];
    }


}

void seq_write_submit(u_int64_t start_lba, u_int64_t num_blocks,uintptr_t dev_addr) {
    //printf("entering task submit!\n");
    
    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        element_per_buffer_write[i] = 0;
        wait_flag_write.emplace_back(threadPool_write.enqueue(thread_runner3, i));
    }

    for(int64_t i = 0; i < num_blocks; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        int64_t dev_id = (start_lba+i) % g_namespaces.size();
        int64_t lba_addr = ((start_lba+i)/ g_namespaces.size())*8 ;//512*8 = 4096
        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        //void* map_addr = dev_addr+embed_entry_width*i ;
        //map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        void* map_addr = devPtr2Map((void*)(dev_addr+i*4UL*1024));
        // void* map_addr = dev_addr+embed_entry_width*i ;
        // map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        //printf("map_addr : %p\n",map_addr);
       //printf("1111\n",i);
        thread_buffer_write[dev_id][element_per_buffer_write[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer_write[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer_write[i][element_per_buffer_write[i]] = std::make_pair(-1, -1);
        ++element_per_buffer_write[i];
    }


}


// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr) {
void cam_gemm_read(u_int64_t * lba_array, u_int64_t req_num,uintptr_t dev_addr) {
    //printf("entering task submit!\n");
    //printf("total_core_num : %d\n",total_core_num);
    for(int32_t i=0; i<(int32_t)total_core_num; ++i) {
        
        wait_flag.emplace_back(threadPool.enqueue(thread_runner_variablecore, i));
    }

    
    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        element_per_buffer[i] = 0;
    }


    for(int64_t i = 0; i < req_num; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        auto [dev_id, lba_addr] = getEmbedAddr(lba_array[i]);
        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        //void* map_addr = dev_addr+embed_entry_width*i ;
        //map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        void* map_addr = devPtr2Map((void*)(dev_addr+embed_entry_width*i));
        //printf("map_addr : %p\n",map_addr);
       //printf("1111\n",i);
        thread_buffer[dev_id][element_per_buffer[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer[i][element_per_buffer[i]] = std::make_pair(-1, -1);
        ++element_per_buffer[i];
    }

    
}

// void task_submit(int64_t embed_num, int32_t *embed_id, void *dev_addr) {
void cam_gemm_write(u_int64_t * lba_array, u_int64_t req_num,uintptr_t dev_addr) {
    //printf("entering task submit!\n");
    for(int32_t i=0; i<(int32_t)total_core_num; ++i) {
        
        wait_flag_write.emplace_back(threadPool_write.enqueue(thread_runner_variablecore_write, i));
    }
    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        element_per_buffer_write[i] = 0;
    }

    for(int64_t i = 0; i < req_num; i++) {
        //printf("**********************\n");
        //printf("p_embed_id[i] : %d\n",p_embed_id[i]);
        auto [dev_id, lba_addr] = getEmbedAddr(lba_array[i]);
        //printf("dev_id : %d\n",dev_id);
        //printf("快吗 : %d \n",lba_addr);
        //printf("i=%d\n",i);
        void* map_addr = devPtr2Map((void*)(dev_addr+embed_entry_width*i));
        // void* map_addr = dev_addr+embed_entry_width*i ;
        // map_addr = (void*)((uintptr_t)map_ptr_base + ((uintptr_t)map_addr - (uintptr_t)dev_ptr_base));
        //printf("map_addr : %p\n",map_addr);
       //printf("1111\n",i);
        thread_buffer_write[dev_id][element_per_buffer_write[dev_id]] = std::make_pair(lba_addr, (int64_t)map_addr);
        //printf("2222\n",i);
        ++element_per_buffer_write[dev_id];
        //printf("3333\n",i);
    }

    for(int32_t i=0; i<(int32_t)g_namespaces.size(); ++i) {
        thread_buffer_write[i][element_per_buffer_write[i]] = std::make_pair(-1, -1);
        ++element_per_buffer_write[i];
    }

    
}