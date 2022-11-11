#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "cuda_runtime.h"
#include <cub/cub.cuh>

#define WARPSIZE 32
#define BLOCKSIZE 1024 

__device__ uint64_t atomicAdd(uint64_t* address, uint64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

__device__ int64_t atomicSub(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) -val);
}

#endif // CUDA_UTILS_H
