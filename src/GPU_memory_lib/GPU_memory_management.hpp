#ifndef GPU_MEMORY_MANAGEMENT
#define GPU_MEMORY_MANAGEMENT

#include <map>
#include <array>
#include <mutex>
#include <functional>

#include <cstdint>
#include <immintrin.h>


class MemCtl {
public:
    virtual ~MemCtl() = default;

    [[nodiscard]] size_t getPoolSize() const {
        return pool_size;
    }

    void *alloc(size_t size);

    void free(void *ptr);

protected:
    MemCtl() = default;

    size_t pool_size{};

    std::mutex allocMutex;
    /*<首地址, 块大小>*/
    std::map<uint64_t, uint64_t> free_chunk, used_chunk;
    /* n_pages, virt_addr_base, phy_addr_array */
    std::tuple<uint32_t, uint64_t, uint64_t *> page_table;
};

class GPUMemCtl : public MemCtl {
public:
    ~GPUMemCtl() override;

    static GPUMemCtl *getInstance(int32_t dev_id, size_t pool_size);
    [[maybe_unused]] static void cleanCtx();
    
protected:
    explicit GPUMemCtl(uint64_t size);

public:
    /*
     * void(uint32_t, uint32_t, uint64_t, uint64_t) => (page_index, page_size, virt_addr, phy_addr)
     */
    void writeTLB(const std::function<void(uint32_t, uint32_t, uint64_t, uint64_t)> &func, bool aggr_flag);

    uint64_t mapV2P(void *ptr);

    void *getDevPtr() const;
    void *getMapDevPtr() const;

    bool chechPhyContiguous() const;

};

#endif