#include "gpu_lcp.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdlib>

#define BLOCK 32

namespace {

#define CHECK(cmd)                                      \
    do {                                                \
        cudaError_t error = (cmd);                      \
        if (cudaSuccess != error) {                     \
            std::cout << __FILE__ << ":" << __LINE__    \
                      << " " << cudaGetErrorString(error) \
                      << std::endl;                     \
            exit(-1);                                   \
        }                                               \
    } while (0)

class _NotUse {
public:
    _NotUse() {
        CHECK(cudaSetDevice(0));
    }
    void func() {}
};

static _NotUse _not_in_user;

class Stream {
public:
    Stream() {
        _not_in_user.func();
        CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, 0));
    }
    ~Stream() {
        CHECK(cudaStreamDestroy(stream_));
    }
    cudaStream_t& get() { return stream_; }
    int get_sm_count() const { return sm_count_; }
    operator cudaStream_t& () {
        return get();
    }
private:
    cudaStream_t stream_{nullptr};
    int sm_count_{0};
};

template <typename T, bool onGPU>
class Memory {
class _Internal {
public:
    explicit _Internal(const int elem_count)
    : elem_count_(elem_count), ptr_(nullptr) {
        if (onGPU) {
            CHECK(cudaMalloc(&ptr_, elem_count_ * sizeof(T)));
        } else {
            CHECK(cudaHostAlloc(&ptr_, elem_count_ * sizeof(T), cudaHostAllocDefault));
        }
        
    }
    ~_Internal() {
        if (onGPU) {
            CHECK(cudaFree(ptr_));
        } else {
            CHECK(cudaFreeHost(ptr_));
        }
        ptr_ = nullptr;
    }
    T* get() { return ptr_; }
    int get_elem_count() const { return elem_count_; }
private:
    const int elem_count_;
    T* ptr_;
};

public:
    explicit Memory()
    : internal_(nullptr) {}
    ~Memory() {}

    void allocate(const int elem_count) {
        if (!(internal_ && internal_->get_elem_count() == elem_count)) {
            internal_.reset(new _Internal(elem_count));
        }
    }

    T* get() {
        if (!internal_) {
            std::cout << "Haven't allocate memory for this object." << std::endl;
            exit(-1);
        } 
        return internal_->get(); 
    }

    operator T*() {
        return get();
    }

    T& operator [](const int i) {
        return get()[i];
    }

private:
    std::unique_ptr<_Internal> internal_;
};
  
} // namespace
/*
namespace Version1 {

__global__ void lcp_kernel(T const* input_string, 
                           int const* sa,
                           int const n,
                           int* const output) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid + 1; i < n; i += stride) {
        int len = 0;
        int a = sa[i], b = sa[i - 1];
        while (input_string[a] == input_string[b]) {
            len++;
            a++;
            b++;
        }
        output[i] = len;
    }
}

void helper(T const* input_string,
            int const* sa,
            int const n, 
            int* const output,
            int const first) {
    static Stream stream;
    static Memory<T, true> d_input;
    d_input.allocate(n);
    static Memory<int, true> d_output;
    d_output.allocate(n);
    static Memory<int, true> d_sa;
    d_sa.allocate(n);

    output[0] = first;

    CHECK(cudaMemcpyAsync(d_input, input_string, sizeof(T) * n, cudaMemcpyDefault, stream.get()));
    CHECK(cudaMemcpyAsync(d_sa, sa, sizeof(int) * n, cudaMemcpyDefault, stream.get()));

    lcp_kernel<<<stream.get_sm_count(), 1024, 0, stream.get()>>>(d_input, d_sa, n, d_output);

    CHECK(cudaMemcpyAsync(output + 1, d_output + 1, sizeof(int) * (n - 1), cudaMemcpyDefault, stream.get()));
    CHECK(cudaStreamSynchronize(stream.get()));
}

} // namespace Version1
*/
inline namespace Version2 {

__global__ void set_flag(uint32_t const* sa,
                         size_t const n,
                         int64_t* flag) {
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = gid + 1; i < n; i += stride) {
        flag[sa[i]] = sa[i - 1];
    }
}

__global__ void compute_plcp(int64_t const *flag,
                             uint32_t const *indices,
                             T const* S,
                             int *plcp,
                             uint32_t const* counter) {
    //for (uint32_t ij = indices[blockIdx.x]; ij < *counter; ij += gridDim.x) {
    {
        uint32_t ij = indices[blockIdx.x];
        uint32_t h = 0;

        if (flag[ij] != -1) {
            int64_t k = flag[ij];
            /*locate first non-identical S*/
            //while (S[ij + h] == S[k + h]) {
            //    h++;
            //}
            int warp_id = 0;
            while (true) {
                uint32_t offset = 32 * warp_id + threadIdx.x;
                uint32_t b_flag = __ballot_sync(__activemask(), S[ij + offset] != S[k + offset]);
                if (b_flag != 0) {
                    int nonzero = __clz(__brev(b_flag));
                    h = 32 * warp_id + nonzero;
                    break;
                } else {
                    warp_id++;
                }
            }
        }
        for (uint32_t l = ij + threadIdx.x; l < indices[blockIdx.x + 1]; l += blockDim.x) {
            plcp[l] = h - (l - ij);
       }
    }
}

__global__ void convert_plcp(int const* plcp,
                             size_t const n,
                             uint32_t const* sa,
                             int *lcp) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < n; i += stride) {
        lcp[i] = plcp[sa[i]];
    }
}

__global__ void indices_flag_kernel(int* indices_flag,
                                    uint32_t* indices,
                                    size_t const n,
                                    T const* input_string,
                                    int64_t const* Flag) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = gid + 1; i < n; i += stride) {
        indices[i] = i;
        if (Flag[i] > 0 && input_string[i-1] != input_string[Flag[i] - 1]) {
            indices_flag[i] = 1;
        }
    }
}

__global__ void update_counter(uint32_t* counter, uint32_t* indices,
                               size_t const n) {
    indices[0] = 0;
    uint32_t old_counter = atomicAdd(counter, 1);
    indices[old_counter + 1] = static_cast<uint32_t>(n);
}

struct ComputeIndicesCpu {
    ComputeIndicesCpu(uint32_t* counter,
                      uint32_t* indices,
                      size_t const n,
                      T const* input_string,
                      int64_t const* Flag,
                      T const* d_input,
                      int const sm_count)
    : counter_(counter), indices_(indices),
    n_(n), input_string_(input_string),
    Flag_(Flag), d_input_(d_input),
    sm_count_(sm_count) {
        std::cout << "Using CPU to compute indices.." << std::endl;
    }
    
    void operator()(cudaStream_t stream) {
        CHECK(cudaStreamSynchronize(stream));
        counter_[0] = 1;
        for (size_t i = 1; i < n_; i++) {
            if (Flag_[i] > 0 && input_string_[i - 1] != input_string_[Flag_[i] - 1]) {
                indices_[counter_[0]++] = i;
            }
        }
        indices_[0] = 0; 
        indices_[counter_[0]] = n_;
    }

    uint32_t* counter_;
    uint32_t* indices_;
    size_t const n_;
    T const* input_string_;
    int64_t const* Flag_;
    T const* d_input_;
    int const sm_count_;
};

struct ComputeIndicesGpu {
    ComputeIndicesGpu(uint32_t* counter,
                      uint32_t* indices,
                      size_t const n,
                      T const* input_string,
                      int64_t const* Flag,
                      T const* d_input,
                      int const sm_count)
    : counter_(counter), indices_(indices),
    n_(n), input_string_(input_string),
    Flag_(Flag), d_input_(d_input),
    sm_count_(sm_count) { 
        std::cout << "Using GPU to compute indices.." << std::endl;

        indices_temp.allocate(n);
        indices_flag.allocate(n);

        cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, indices_temp.get(),
                                indices_flag.get(), indices_ + 1,
                                counter_, n);
        temp.allocate(temp_storage_bytes);
    }

    void operator()(cudaStream_t stream) {
        CHECK(cudaMemsetAsync(indices_temp, 0, sizeof(uint32_t) * n_, stream));
        CHECK(cudaMemsetAsync(indices_flag, 0, sizeof(int) * n_, stream));

        indices_flag_kernel<<<sm_count_, BLOCK, 0, stream>>>(
                    indices_flag, indices_temp, n_, d_input_, Flag_);

        cub::DeviceSelect::Flagged(temp.get(), temp_storage_bytes, indices_temp.get(),
                               indices_flag.get(), indices_ + 1,
                               counter_, n_, stream);

        CHECK(cudaStreamSynchronize(stream));
        indices_[0] = 0;
        indices_[++counter_[0]] = n_;
        //update_counter<<<1, 1, 0, stream>>>(counter_, indices_, n_);
    }

    uint32_t* counter_;
    uint32_t* indices_;
    size_t const n_;
    T const* input_string_;
    int64_t const* Flag_;
    T const* d_input_;
    int const sm_count_;

    Memory<uint32_t, true> indices_temp;
    Memory<int, true> indices_flag;
    size_t temp_storage_bytes = 0;
    Memory<T, true> temp;
};

#define ComputeIndices ComputeIndicesGpu

void helper(T const* input_string,
            uint32_t const* sa,
            size_t const n, 
            int* const output,
            int const first) { 
    static Stream stream;
    static Memory<T, true> d_input;
    d_input.allocate(n);
    static Memory<int, true> d_output;
    d_output.allocate(n);
    static Memory<uint32_t, true> d_sa;
    d_sa.allocate(n);
    static Memory<int64_t, false> Flag;
    Flag.allocate(n);
    static Memory<uint32_t, false> indices;
    indices.allocate(n);
    static Memory<int, false> plcp;
    plcp.allocate(n);
    static Memory<uint32_t, false> counter;
    counter.allocate(1);

    ComputeIndices compute_indices(counter, indices, n, input_string, 
                                   Flag, d_input, stream.get_sm_count());
    // set first elem of LCP
    output[0] = first;

    CHECK(cudaMemcpyAsync(d_input, input_string, sizeof(T) * n, cudaMemcpyDefault, stream.get()));
    CHECK(cudaMemcpyAsync(d_sa, sa, sizeof(uint32_t) * n, cudaMemcpyDefault, stream.get()));
    CHECK(cudaMemsetAsync(plcp, 0, sizeof(int) * n, stream.get()));

    const auto begin = std::chrono::high_resolution_clock::now();

    // compute flag array
    Flag[sa[0]] = -1;
    set_flag<<<stream.get_sm_count(), BLOCK, 0, stream.get()>>>(d_sa, n, Flag);

    // compute indices
    compute_indices(stream.get());
    
    // compute PLCP array
    size_t grid = counter[0];
    compute_plcp<<<grid, 32, 0, stream.get()>>>(Flag, indices, d_input, plcp, counter);

    // convert PLCP to LCP
    convert_plcp<<<stream.get_sm_count(), BLOCK, 0, stream.get()>>>(plcp, n, d_sa, d_output);

    CHECK(cudaMemcpyAsync(output + 1, d_output + 1, sizeof(int) * (n - 1), cudaMemcpyDefault, stream.get()));
    CHECK(cudaStreamSynchronize(stream.get()));

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::ratio<1, 1>> elapsed(end - begin);
    std::cout << "gpu LCP took " << elapsed.count() << " seconds to finish." << std::endl;
}


} // namespace Version2

namespace Version3 {

template <typename Type>
__device__ void swap(Type& a, Type& b) {
    Type temp = a;
    a = b;
    b = temp;
}

__device__ int32_t match_length(T const* beginInput,
                                T const* endInput,
                                int32_t indexA,
                                int32_t indexB,
                                int32_t matchLength) {
    if (indexA > indexB) 
        swap(indexA, indexB);

    T const * inputA = beginInput + indexA + matchLength;
    T const * inputB = beginInput + indexB + matchLength;
    while ((inputA < endInput) && (*inputA == *inputB))
        ++inputA, ++inputB;
    return (inputA - (beginInput + indexA));
}

__device__ void lcp_per_thread(T const* beginInput,
                               T const* endInput,
                               int32_t* begin,
                               size_t size,
                               size_t currentMatchLength) {
    if (size <= 1) {
        for (auto i = 0; i < size; ++i) {
            begin[i] = match_length(beginInput, endInput, begin[i], begin[i+1], currentMatchLength);
        }
    } else {
        auto mid = (size / 2);
        auto nextMatchLength = match_length(beginInput, endInput, begin[0], begin[mid], currentMatchLength);
        lcp_per_thread(beginInput, endInput, begin, mid, nextMatchLength);
        lcp_per_thread(beginInput, endInput, begin + mid, size - mid, currentMatchLength);
    }   
}

__global__ void lcp_kernel_1(T const* beginInput,
                             T const* endInput,
                             int32_t* begin,
                             size_t size,
                             int32_t* temp) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numThreads = blockDim.x * gridDim.x;
    size_t perThread = (size + numThreads - 1) / numThreads;

    if (gid < size) {
        auto s = perThread;
        auto offset = gid * perThread;
        if (offset + s > size) s = size - offset;
        temp[gid] = match_length(beginInput, endInput, begin[offset + s - 1], begin[offset + s], 0);
        lcp_per_thread(beginInput, endInput, begin + offset, s - 1, 0);
    }
}

__global__ void lcp_kernel_2(size_t size, 
                             int32_t* begin,
                             int32_t const* temp) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numThreads = blockDim.x * gridDim.x;
    size_t perThread = (size + numThreads - 1) / numThreads;

    if (gid < size) {
        auto s = perThread;
        auto offset = gid * perThread;
        if (offset + s > size) s = size - offset;
        begin[offset + s - 1] = temp[gid];
    }
}

void helper(T const* input_string,
            int const* sa,
            int const n,
            int* const output,
            int const first) {
    static Stream stream;

    static Memory<int, true> d_output;
    d_output.allocate(n);
    CHECK(cudaMemcpyAsync(d_output, sa, sizeof(int) * (n), cudaMemcpyDefault, stream));

    static Memory<T, true> d_input;
    d_input.allocate(n);
    CHECK(cudaMemcpyAsync(d_input, input_string, sizeof(T) * n, cudaMemcpyDefault, stream));

    static Memory<int, true> d_sa;
    d_sa.allocate(n);
    CHECK(cudaMemcpyAsync(d_sa, sa, sizeof(int) * n, cudaMemcpyDefault, stream));

    static Memory<int, true> temp;
    temp.allocate(n);
    CHECK(cudaMemsetAsync(temp, 0, sizeof(int) * n, stream));

    const auto begin = std::chrono::high_resolution_clock::now();

    constexpr size_t block = 1;
    size_t grid = ((n-1) + block - 1) / block;
    lcp_kernel_1<<<grid, block, 0, stream>>>(/*beginInput=*/d_input,
                                             /*endInput=*/d_input + n,
                                             /*begin=*/d_output,
                                             /*size=*/n - 1,
                                             /*temp=*/temp);
    CHECK(cudaStreamSynchronize(stream));
    std::cout << __LINE__ << " here" << std::endl;

    lcp_kernel_2<<<grid, block, 0, stream>>>(/*size=*/n - 1,
                                             /*begin=*/d_output,
                                             /*temp=*/temp);

    output[0] = first;
    CHECK(cudaMemcpyAsync(output + 1, d_output, sizeof(int) * (n - 1), cudaMemcpyDefault, stream));
    CHECK(cudaStreamSynchronize(stream));

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::ratio<1, 1>> elapsed(end - begin);
    std::cout << "gpu LCP took " << elapsed.count() << " seconds." << std::endl;
}


} // namespace Version3

namespace CPUVersion {


    auto match_length(int8_t const* beginInput,
                           int8_t const* endInput,
                           int32_t indexA,
                           int32_t indexB,
                           int32_t matchLength) -> int32_t {
        if (indexA > indexB) std::swap(indexA, indexB);

        int8_t const * inputA = beginInput + indexA + matchLength;
        int8_t const * inputB = beginInput + indexB + matchLength;
        endInput -= sizeof(int64_t);
        while ((inputB < endInput) && 
               (*(int64_t const*)inputA == *(int64_t const*)inputB))
            inputA += sizeof(int64_t), inputB += sizeof(int64_t);
        endInput += sizeof(int64_t);
        while ((inputB < endInput) && (*inputA == *inputB)) 
            ++inputA, ++inputB;
        return (inputA - (beginInput + indexA));
    };

    auto lcp(int8_t const * beginInput, 
                  int8_t const * endInput,
                  int32_t* begin,
                  size_t size,
                  size_t currentMatchLength) -> void {
        if (size <= 4) {
            for (auto i = 0; i < size; ++i) 
                begin[i] = match_length(beginInput, endInput,
                                        begin[i], begin[i+1],
                                        currentMatchLength);
        } else {
            auto mid = (size / 2);
            auto nextMatchLength = match_length(beginInput, endInput, begin[0], begin[mid], currentMatchLength);
            lcp(beginInput, endInput, begin, mid, nextMatchLength);
            lcp(beginInput, endInput, begin + mid, size - mid, currentMatchLength);
        }
    };

    auto lcp_multi(T const* beginInput, 
                        T const* endInput,
                        int32_t * begin,
                        size_t const size,
                        int32_t numThreads) -> void {
         auto perThread = (size + numThreads - 1) / numThreads;
         std::vector<std::thread> threads(numThreads);
         std::unique_ptr<int []> temp(new int[numThreads]());

         auto n = 0;
         for (auto i = 0; i < numThreads; i++) {
            auto s = perThread;
            if ((n + s) > size) s = (size - n);

            temp[i] = match_length((int8_t const *)beginInput, 
                                    (int8_t const *)endInput, begin[n + s - 1], 
                                    begin[n + s], 0);
            threads[i] = std::thread(lcp, (int8_t const*)beginInput, 
                                     (int8_t const*)endInput, 
                                     begin + n, s - 1, 0);
            n += s;
         }

         for (auto &t : threads) t.join();

         n = 0;
         for (auto i = 0; i < numThreads; i++) {
            auto s = perThread;
            if ((n + s) > size) s = (size - n);
            begin[n + s - 1] = temp[i];
            n += s;  
         }
    };

int GetNumThreads() {
    const auto num_threads_env = std::getenv("CpuNumThreads");
    int num_threads = 0;
    if (nullptr == num_threads_env) {
        num_threads = std::thread::hardware_concurrency();
    } else {
        num_threads = std::atoi(num_threads_env);
    }
    std::cout << "numThreads = " << num_threads << std::endl;
    return num_threads;
}

void helper(T const* input_string,
            int const* sa,
            int const n,
            int* const output,
            int const first) {
    auto const begin = std::chrono::high_resolution_clock::now();

    output[0] = first;
    std::memcpy(output + 1, sa, sizeof(int) * (n));

    lcp_multi(input_string, input_string + n, 
              output + 1, n - 1, 
              /*numThreads=*/GetNumThreads());

    auto const end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::ratio<1, 1>> elapsed(end - begin);
    std::cout << "CPU LCP took " << elapsed.count() << " seconds" << std::endl;
}

} // namespace CPUVersion

void gpuLcpArray(T const* input_string,
                 uint32_t const* sa,
                 size_t const n, 
                 int* const output,
                 int const first) {
    CPUVersion::helper(input_string, (int32_t const*)sa, n, output, first);
}
