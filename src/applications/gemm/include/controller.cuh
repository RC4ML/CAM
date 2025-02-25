#ifndef __AEOLUS_CONTROLLER_CUH
#define __AEOLUS_CONTROLLER_CUH

#include <vector>
#include "device.cuh"
#include "request.cuh"

enum aeolus_access_type
{
    AEOLUS_ACCESS_SEQUENTIAL  = 0,
    AEOLUS_ACCESS_RANDOM      = 1
};

enum aeolus_dist_type
{
    AEOLUS_DIST_STRIPE  = 0,
    AEOLUS_DIST_REPLICA = 1
};

enum aeolus_buf_type
{
    AEOLUS_BUF_USER   = 0,
    AEOLUS_BUF_PINNED = 1
};

/**
 * @brief A controller manages multiple IO queues of multiple SSDs and provides a simple interface for user.
 * 
 */
class Controller
{
protected:
    std::vector<Device*>ssd_list;
    int                 ssd_count;
    int                 gpu_id;
    int32_t             max_io_size;
    int32_t             num_queue_per_ssd;
    int32_t             queue_depth;
    aeolus_dist_type  dist_type;
    aeolus_buf_type   buf_type;
    
    int32_t             max_queue_num;
    int32_t             max_trans_size;

    int32_t             **qpid_list;
    IoQueuePair         *d_ssdqp;
    void                *d_iobuf_ptr;
    uint64_t            *d_iobuf_phys;
    uint64_t            *prp_list;
    uint64_t            *h_prp_phys;
    uint64_t            *d_prp_phys;
    uint64_t            *h_ssd_num_lbs;
    uint64_t            *d_ssd_num_lbs;

    aeolus_dev_mem_context *qp_ctx;
    aeolus_dev_mem_context *iobuf_ctx;

    int* ssd_num_reqs;
    Request *distributed_reqs;
    int *req_ids;
public:
    uint64_t max_lb_number;
    uint64_t *qp_phys;

    /**
     * @brief Construct a new Controller object. A controller manages multiple IO queues of multiple SSDs and provides a simple interface for user.
     * 
     * @param ssd_list List of SSD devices to be managed by the controller.
     * @param num_queue_per_ssd Number of IO queues allocated to each SSD.
     * @param max_io_size Maximum IO size in bytes in a single NVMe command.
     * @param queue_depth Depth of each IO queue.
     * @param dist_type Pattern for data distribution. AEOLUS_DIST_STRIPE means data is striped across SSDs, AEOLUS_DIST_REPLICA means data is replicated across SSDs.
     * @param buf_type Type of data buffer for IO. AEOLUS_BUF_USER means the buffer can be arbitrary buffer provided by user, AEOLUS_BUF_PINNED means the buffer is pinned by user beforehand.
     * @param pinned_buf_phys Physical addresses of the pinned buffer. Only valid when buf_type is AEOLUS_BUF_PINNED.
     * @param pinned_buf_size Size of the pinned buffer. Only valid when buf_type is AEOLUS_BUF_PINNED.
     */
    Controller(
        std::vector<Device*> ssd_list, 
        int32_t num_queue_per_ssd    = AEOLUS_MAX_NUM_QUEUES,
        int32_t max_io_size          = 4096,
        int32_t queue_depth          = 4096,
        aeolus_dist_type dist_type = AEOLUS_DIST_STRIPE,
        aeolus_buf_type  buf_type  = AEOLUS_BUF_USER,
        uint64_t *pinned_buf_phys = nullptr,
        uint64_t pinned_buf_size = 0
    );

    /**
     * @brief Construct a new Controller object. This interface hides the details of queue depth and IO size and provides pre-defined configurations to user.
     * 
     * @param ssd_list List of SSD devices to be managed by the controller.
     * @param access_type Preset of IO pattern. AEOLUS_ACCESS_SEQUENTIAL means the user prefers sequential access, AEOLUS_ACCESS_RANDOM means random access.
     * @param dist_type Pattern for data distribution. AEOLUS_DIST_STRIPE means data is striped across SSDs, AEOLUS_DIST_REPLICA means data is replicated across SSDs.
     * @param buf_type Type of data buffer for IO. AEOLUS_BUF_USER means the buffer can be arbitrary buffer provided by user, AEOLUS_BUF_PINNED means the buffer is pinned by user beforehand.
     */
    inline Controller(
        std::vector<Device*> ssd_list, 
        aeolus_access_type access_type = AEOLUS_ACCESS_SEQUENTIAL,
        aeolus_dist_type   dist_type   = AEOLUS_DIST_STRIPE,
        aeolus_buf_type    buf_type    = AEOLUS_BUF_USER
    ) : Controller(
        ssd_list,
        access_type == AEOLUS_ACCESS_SEQUENTIAL ? 8 : AEOLUS_MAX_NUM_QUEUES,
        access_type == AEOLUS_ACCESS_SEQUENTIAL ? AEOLUS_MAX_DATA_TRANSFER : 4096,
        access_type == AEOLUS_ACCESS_SEQUENTIAL ? 256 : 4096,
        dist_type,
        buf_type
    )
    {};

    ~Controller();
    void read_data(uint64_t start_lb, uint64_t num_lb, void *buf);
    void write_data(uint64_t start_lb, uint64_t num_lb, void *buf);
    IoQueuePair *get_io_queue_pair() { return d_ssdqp; }

    /**
     * @brief Submit a batch of IO requests to the controller and process them in a helper thread
     * until completion.
     * 
     * @param req List of IO requests.
     * @param num_req Number of IO requests.
     * @param dir Direction of requests. All requests must have the same direction.
     * @param stream cuda stream used.
     * @param d_prp_phys Physical addresses of the PRP list. Only valid when buf_type is AEOLUS_BUF_PINNED. (optional)
     */
    virtual void submit_io_req(Request *req, int num_req, aeolus_access_dir dir, cudaStream_t stream, uint64_t *d_prp_phys = nullptr) = 0;

private:
    void lb_to_ssd_id(uint64_t lb, int &ssd_id, uint64_t &local_lb);
};

/**
 * @brief A controller instance with helper thread-based IO processing functions.
 * requests are processed in multiple batches. 
 * 
 */
class ControllerLegacy : public Controller
{
public:
    inline ControllerLegacy(
        std::vector<Device*> ssd_list, 
        int32_t num_queue_per_ssd    = AEOLUS_MAX_NUM_QUEUES,
        int32_t max_io_size          = 4096,
        int32_t queue_depth          = 4096,
        aeolus_dist_type dist_type = AEOLUS_DIST_STRIPE,
        aeolus_buf_type  buf_type  = AEOLUS_BUF_USER,
        uint64_t *pinned_buf_phys = nullptr,
        uint64_t pinned_buf_size = 0
    ) : Controller(
        ssd_list,
        num_queue_per_ssd,
        max_io_size,
        queue_depth,
        dist_type,
        buf_type,
        pinned_buf_phys,
        pinned_buf_size
    )
    {};

    /**
     * @brief Submit a batch of IO requests to the controller and process them in a helper thread
     * until completion.
     * 
     * @param req List of IO requests.
     * @param num_req Number of IO requests.
     * @param dir Direction of requests. All requests must have the same direction.
     * @param stream cuda stream used.
     * @param d_prp_phys Physical addresses of the PRP list. Only valid when buf_type is AEOLUS_BUF_PINNED. (optional)
     */
    void submit_io_req(Request *req, int num_req, aeolus_access_dir dir, cudaStream_t stream, uint64_t *d_prp_phys = nullptr) override;
};

/**
 * @brief A controller instance with submit-poll processing interface.
 * @warning Each instance can only process one batch of request at one time,
 * and that the number of requests in a batch should not exceed the queue number times queue depth.
 * 
 */
class ControllerDecoupled : public Controller
{
private:
    int* ssd_num_reqs_prefix_sum;
    int num_reqs;
    cudaStream_t stream;
    aeolus_access_dir dir;
public:
    inline ControllerDecoupled(
        std::vector<Device*> ssd_list, 
        int32_t num_queue_per_ssd    = AEOLUS_MAX_NUM_QUEUES,
        int32_t max_io_size          = 4096,
        int32_t queue_depth          = 4096,
        aeolus_dist_type dist_type = AEOLUS_DIST_STRIPE,
        aeolus_buf_type  buf_type  = AEOLUS_BUF_USER,
        uint64_t *pinned_buf_phys = nullptr,
        uint64_t pinned_buf_size = 0
    ) : Controller(
        ssd_list,
        num_queue_per_ssd,
        max_io_size,
        queue_depth,
        dist_type,
        buf_type,
        pinned_buf_phys,
        pinned_buf_size
    )
    {
        AEOLUS_CUDA_CHECK(cudaMalloc(&ssd_num_reqs_prefix_sum, ssd_count * sizeof(int)));
    }

    /**
     * @brief Submit a batch of IO requests to the NVMe SSDs. User needs to ensure the completion
     * of the requests by `poll()` function.
     * 
     * @param req List of IO requests.
     * @param num_req Number of IO requests.
     * @param dir Direction of requests. All requests must have the same direction.
     * @param stream cuda stream used.
     * @param d_prp_phys Physical addresses of the PRP list. Only valid when buf_type is AEOLUS_BUF_PINNED. (optional)
     * 
     * @warning User must make sure that number of requests is no greater than the queue number times queue depth.
     */
    void submit_io_req(Request *req, int num_req, aeolus_access_dir dir, cudaStream_t stream, uint64_t* d_prp_phys = nullptr) override;

    /**
     * @brief Poll the in-flight requests until completion.
     * 
     * @warning User must not submit new requests before the completion of the previous batch.
     */
    void poll();
};

#endif