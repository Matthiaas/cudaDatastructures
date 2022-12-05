#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "cuda_runtime.h"
#include <cub/cub.cuh>


#define _CG_ABI_EXPERIMENTAL
#define _CG_CPP11_FEATURES
#include <cooperative_groups.h>


static __device__ __forceinline__ uint64_t atomicAdd(uint64_t* address, uint64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

static __device__ __forceinline__ int64_t atomicAdd(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

static __device__ __forceinline__ int64_t atomicSub(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) -val);
}

static __device__ __forceinline__ uint64_t atomicCAS(uint64_t* address, uint64_t cmp, uint64_t val) {
    return ::atomicCAS((unsigned long long*) address, (unsigned long long) val, (unsigned long long) val);
}

static __device__ __forceinline__ int64_t atomicCAS(int64_t* address, int64_t cmp, uint64_t val) {
    return ::atomicCAS((unsigned long long*) address, (unsigned long long) val, (unsigned long long) val);
}


// https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
template<typename T>
static __device__ __forceinline__ T atomicAggInc(T *ctr) {
  auto active = cooperative_groups::coalesced_threads();
  T warp_res;
  if(active.thread_rank() == 0)
    warp_res = atomicAdd(ctr, active.size());
  return active.shfl(warp_res, 0) + active.thread_rank();
}

#endif // CUDA_UTILS_H
