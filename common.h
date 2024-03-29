#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#define _CG_ABI_EXPERIMENTAL
#define _CG_CPP11_FEATURES
#include <cooperative_groups.h>

#include <type_traits>

enum Platform {
    CPU,
    GPU
};

 #ifdef __CUDA_ARCH__
    constexpr Platform CurrentPlatform = Platform::GPU;
#else
    constexpr Platform CurrentPlatform = Platform::CPU;
#endif



template <typename T, typename Inc>
__device__ __host__ __forceinline__ T fetch_and_add(T* elment, Inc inc) {
    if constexpr (CurrentPlatform== Platform::CPU) {
        return elment->fetch_add(inc);
    } else {
        return atomicAdd(elment, inc);
    }
}


__device__ __host__ __forceinline__ void platformMemFence() {
#ifdef __CUDA_ARCH__
    __threadfence();
#else
    __sync_synchronize();
#endif
}

__device__ __host__ __forceinline__ unsigned platformCAS(
        unsigned* address, unsigned compare, unsigned val) {
    if constexpr (CurrentPlatform == Platform::CPU) {
        return __sync_val_compare_and_swap(address, compare, val);
    } else {
        return atomicCAS(address, compare, val);
    }
}

__device__ __host__ __forceinline__ unsigned long long platformCAS(
        unsigned long long* address, unsigned long long compare, 
        unsigned long long val) {
    if constexpr (CurrentPlatform == Platform::CPU) {
        return __sync_val_compare_and_swap(address, compare, val);
    } else {
        return atomicCAS(address, compare, val);
    }
}


#endif
