#ifndef ATOMICADD_H
#define ATOMICADD_H

#include <cinttypes>
#include "cuda_runtime.h"

namespace atomicadd {

template <typename T>
__global__ void add_as_accumuluated_requests(T *v, uint32_t iters);

template <typename T>
__global__ void add_as_requests(T *v, uint32_t iters);

template <typename T>
__global__ void add_warp_inc(T *v, uint32_t iters);

template <typename T>
__global__ void add_trival(T *v, uint32_t iters);


}

#endif