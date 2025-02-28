#include "GPU_memory_management.hpp"


#include <map>
#include <memory>
#include <string>
#include <string_view>

// #include <fmt/core.h>
// #include <fmt/chrono.h>
// #include <fmt/ranges.h>
// #include <fmt/os.h>
// #include <fmt/args.h>
// #include <fmt/ostream.h>
// #include <fmt/std.h>	
// #include <fmt/color.h>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <immintrin.h>

#include <rc4ml.h>



#include <cuda.h>
#include <gdrapi.h>

#define ASSERT(x)                                               \
    do                                                          \
    {                                                           \
        if (!(x))                                               \
        {                                                       \
            fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

#define ASSERTDRV(stmt)                     \
    do                                      \
    {                                       \
        CUresult result = (stmt);           \
        if (result != CUDA_SUCCESS) {       \
            const char *_err_name;          \
            cuGetErrorName(result, &_err_name); \
            fprintf(stderr, "CUDA error: %s\n", _err_name); \
        }                                   \
        ASSERT(CUDA_SUCCESS == result);     \
    } while (0)

#define ASSERT_EQ(P, V) ASSERT((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT(!((P) == (V)))



[[maybe_unused]] static bool debug_flag = false;



// [[maybe_unused]] static void errorPrint(std::string_view str) {
// 	fmt::print(fg(fmt::color::red), "{}\n", str);
// }

// [[maybe_unused]] static void passPrint(std::string_view str) {
// 	fmt::print(fg(fmt::color::green), "{}\n", str);
// }

// [[maybe_unused]] static void warnPrint(std::string_view str) {
// 	fmt::print(fg(fmt::color::yellow), "{}\n", str);
// }

// [[maybe_unused]] static void infoPrint(std::string_view str) {
// 	fmt::print(fg(fmt::color::cyan), "{}\n", str);
// }

static const size_t config_region_size = 256*1024;
static const size_t lite_region_size = 4*1024;
static const size_t bridge_region_size = 1024*1024*1024;

std::pair<uint64_t, uint64_t> findFreeChunk(const std::map<uint64_t, uint64_t> &freeChunk, uint64_t mSize) {
    for (auto const &it: freeChunk) {
        if (it.second >= mSize) {
            return {it.first, it.second};
        }
    }
    return {0, 0};
}

static bool contains(const std::map<uint64_t, uint64_t> &mp, uint64_t addr) {
    auto it = mp.find(addr);
    if (it == mp.end()) {
        return false;
    } else {
        return true;
    }
}

void *MemCtl::alloc(size_t size) {
    size = (size + 64UL - 1) & ~(64UL - 1);
    std::lock_guard<std::mutex> lock(allocMutex);
    /*查找大小大于申请空间大小的空闲内存块*/
    auto ck = findFreeChunk(free_chunk, size);
    auto &free_addr = ck.first;
    auto &free_size = ck.second;
    /*如果找到的块为空则报告申请失败*/
    if (free_addr == 0) {
       // warnPrint(fmt::format("No Free CPU Chunk. Alloc failed!"));
        return nullptr;
    }
    /*如果内存块分配后仍存在剩余空间, 从内存块高地址部分分配*/
    if (free_size > size) {
        free_chunk[free_addr] = free_size - size;
        used_chunk[free_addr + free_size - size] = size;
        return (void *) (free_addr + free_size - size);
    } else {
        free_chunk.erase(free_addr);
        used_chunk[free_addr] = size;
        return (void *) (free_addr);
    }
}

void MemCtl::free(void *ptr) {
    std::lock_guard<std::mutex> lock(allocMutex);
    /*检查释放的内存块的合法性*/
    if (!contains(used_chunk, (uint64_t) ptr)) {
       // errorPrint(fmt::format("Pointer to free is not in Alloc Log"));
        exit(1);
    }
    auto it = used_chunk.find((uint64_t) ptr);
    uint64_t free_size = it->second;
    used_chunk.erase(it);
    /*寻找第一个首地址大于ptr的空闲块, 返回map结构的迭代器*/
    auto nextIt = free_chunk.upper_bound((uint64_t) ptr);
    if (!free_chunk.empty()) {
        auto prevIt = std::prev(nextIt);
        /*检查前置空闲块 首地址+块大小 与 释放块首地址 是否连续, 连续则将释放块合并到前置空闲块中*/
        if (prevIt->first + prevIt->second == (uint64_t) ptr) {
            free_size += prevIt->second;
            ptr = (void *) prevIt->first;
        }
    }
    /*合并后置块*/
    if (nextIt != free_chunk.end() && (uint64_t) ptr + free_size == nextIt->first) {
        free_size += nextIt->second;
        free_chunk.erase(nextIt);
    }
    free_chunk[(int64_t) ptr] = free_size;
}



class gdrMemAllocator {
public:
    ~gdrMemAllocator();

    CUresult gpuMemAlloc(CUdeviceptr *pptr, size_t psize, bool align_to_gpu_page = true, bool set_sync_memops = true);

    CUresult gpuMemFree(CUdeviceptr pptr);

private:
    std::map<CUdeviceptr, CUdeviceptr> _allocations;
};

gdrMemAllocator::~gdrMemAllocator() {
    for (auto &it: _allocations) {
        CUresult ret;
        ret = cuMemFree(it.second);
        if (ret != CUDA_SUCCESS) {
           // warnPrint(fmt::format("Fail to free cuMemAlloc GPU Memory"));
        }
    }
}

CUresult gdrMemAllocator::gpuMemAlloc(CUdeviceptr *pptr, size_t psize, bool align_to_gpu_page, bool set_sync_memops) {
    CUresult ret = CUDA_SUCCESS;
    CUdeviceptr ptr;
    size_t size;

    if (align_to_gpu_page) {
        size = psize + GPU_PAGE_SIZE - 1;
    } else {
        size = psize;
    }

    ret = cuMemAlloc(&ptr, size);
    if (ret != CUDA_SUCCESS)
        return ret;

    if (set_sync_memops) {
        unsigned int flag = 1;
        ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
        if (ret != CUDA_SUCCESS) {
            cuMemFree(ptr);
            return ret;
        }
    }

    if (align_to_gpu_page) {
        *pptr = (ptr + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    } else {
        *pptr = ptr;
    }
    // Record the actual pointer for doing gpuMemFree later.
    _allocations[*pptr] = ptr;

    return CUDA_SUCCESS;
}

CUresult gdrMemAllocator::gpuMemFree(CUdeviceptr pptr) {
    CUresult ret = CUDA_SUCCESS;
    CUdeviceptr ptr;

    if (_allocations.count(pptr) > 0) {
        ptr = _allocations[pptr];
        ret = cuMemFree(ptr);
        if (ret == CUDA_SUCCESS)
            _allocations.erase(pptr);
        return ret;
    } else {
        return CUDA_ERROR_INVALID_VALUE;
    }
}

static gdrMemAllocator allocator;

static int32_t devID{-1};

static const gdr_mh_t null_mh = {0};

static gdr_t gdrDev{};
static gdr_mh_t gdrUserMapHandler{null_mh};
static gpu_tlb_t gdrPageTable{};
static gdr_info_t info{};

static CUdeviceptr devAddr{};
static void *mapDevPtr{};

static inline bool operator==(const gdr_mh_t &a, const gdr_mh_t &b) {
    return a.h == b.h;
}


static std::vector<std::shared_ptr<GPUMemCtl>> gpu_mem_ctl_list;

GPUMemCtl::GPUMemCtl([[maybe_unused]]uint64_t size) {

    pool_size = size;
    auto page_size = 64UL * 1024;

    CUdevice dev;
    CUcontext devCtx;
    ASSERTDRV(cuInit(0));
    ASSERTDRV(cuDeviceGet(&dev, devID));
    ASSERTDRV(cuDevicePrimaryCtxRetain(&devCtx, dev));
    ASSERTDRV(cuCtxSetCurrent(devCtx));

    ASSERTDRV(allocator.gpuMemAlloc(&devAddr, size));

    gdrDev = gdr_open();
    ASSERT_NEQ(gdrDev, nullptr);

    // 64KB * 64K = 4GB
    // 4GB * 20 = 80GB
    gdrPageTable.pages = new uint64_t[65536 * 20];

    ASSERT_EQ(rc4ml_pin_buffer(gdrDev, devAddr, size, 0, 0, &gdrUserMapHandler, &gdrPageTable), 0);
    ASSERT_NEQ(gdrUserMapHandler, null_mh);

    ASSERT_EQ(gdr_map(gdrDev, gdrUserMapHandler, &mapDevPtr, size), 0);

    ASSERT_EQ(gdr_get_info(gdrDev, gdrUserMapHandler, &info), 0);

    ASSERT_EQ((info.va - devAddr), 0);
    ASSERT_EQ((devAddr & (page_size - 1)), 0);

    page_table = {gdrPageTable.page_entries, (uint64_t) (devAddr), gdrPageTable.pages};
    free_chunk.emplace((uint64_t) devAddr, size);

}

GPUMemCtl::~GPUMemCtl() {

    const auto size = std::get<0>(page_table) * 64UL * 1024;
    delete[] std::get<2>(page_table);
    ASSERT_EQ(gdr_unmap(gdrDev, gdrUserMapHandler, mapDevPtr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(gdrDev, gdrUserMapHandler), 0);
    ASSERT_EQ(gdr_close(gdrDev), 0);
    ASSERTDRV(allocator.gpuMemFree(devAddr));

}

GPUMemCtl *GPUMemCtl::getInstance([[maybe_unused]]int32_t dev_id, [[maybe_unused]]size_t pool_size) {

    if (devID >= 0 && devID != dev_id) {
        // errorPrint(fmt::format("This QDMA library now only support one GPU Memory Pool"));
        // errorPrint(fmt::format("New device id {} is not equal to previous device id {}", dev_id, devID));
        exit(1);
    }
    // up round to 64KB
    pool_size = (pool_size + 64UL * 1024 - 1) & ~(64UL * 1024 - 1);

    if (pool_size % (2UL * 1024 * 1024) != 0) {
        // warnPrint(fmt::format("Suggest GPU Memory Pool Size to be multiple of 2MB for Page Aggregation"));
        // errorPrint(fmt::format("For correctness safety, the program will exit. Please change the pool size"));
        exit(1);
    }

    if (gpu_mem_ctl_list.empty()) {
        devID = dev_id;
        auto tmp = new GPUMemCtl(pool_size);
        gpu_mem_ctl_list.push_back(std::shared_ptr<GPUMemCtl>(tmp));
        return tmp;
    } else {
        static bool warn_flag = false;
        if (!warn_flag) {
            warn_flag = true;
            // warnPrint(fmt::format("This QDMA library now only support one GPU Memory Pool"));
            // warnPrint(fmt::format("Request pool size will be ignored"));
            // warnPrint(fmt::format("The previous GPU Memory Pool with size {} will be returned",
            //                       gpu_mem_ctl_list[0]->getPoolSize()));
        }
        return gpu_mem_ctl_list[0].get();
    }

}

void GPUMemCtl::cleanCtx() {

    gpu_mem_ctl_list.clear();


}

void GPUMemCtl::writeTLB([[maybe_unused]]const std::function<void(uint32_t, uint32_t, uint64_t, uint64_t)> &func, [[maybe_unused]]bool aggr_flag) {

    const auto &[n_pages, vaddr, parray] = page_table;

    if (aggr_flag) {
        const auto page_size = 2UL * 1024 * 1024;
        auto aggr_n_pages = n_pages / 32;
        for (uint32_t i = 0; i < aggr_n_pages; ++i) {
            for (uint32_t j = 1; j < 32; ++j) {
                ASSERT_EQ((parray[i * 32 + j] - parray[i * 32 + j - 1]), 65536);
            }
            func(i, page_size, vaddr + i * page_size, parray[i * 32]);
        }
    } else {
        const auto page_size = 64UL * 1024;
        for (int i = 0; i < n_pages; i++) {
            func(i, page_size, vaddr + i * page_size, parray[i]);
        }
    }

}

uint64_t GPUMemCtl::mapV2P(void *ptr) {
    const auto &[n_pages, vaddr, parray] = page_table;
    const auto page_size = 64UL * 1024;
    uint64_t offset = (uint64_t) ptr - vaddr;
    return parray[offset / page_size] + (offset & (page_size - 1));
}

void *GPUMemCtl::getDevPtr() const {

    return (void *)devAddr;

}

void *GPUMemCtl::getMapDevPtr() const {

    return mapDevPtr;

}

bool GPUMemCtl::chechPhyContiguous() const {

    const auto &[n_pages, vaddr, parray] = page_table;
    const auto page_size = 64UL * 1024;
    for (int i = 1; i < n_pages; i++) {
        if (parray[i] - parray[i - 1] != page_size) {
            return false;
        }
    }
    return true;

}
