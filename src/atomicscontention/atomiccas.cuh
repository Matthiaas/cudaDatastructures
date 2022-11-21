#ifndef ATOMICCAS_H
#define ATOMICCAS_H

#include "cuda_runtime.h"
#include <cinttypes>

#define WARPSIZE 32
#define BLOCKSIZE 1024

namespace atomiccas {

template <typename T>
__global__ void add_as_accumuluated_requests(T *v, uint32_t iters);

template <typename T>
__global__ void add_as_requests(T *v, uint32_t iters);

template <typename T>
__global__ void add_trival(T *v, uint32_t iters);

}

#endif // ATOMICCAS_H