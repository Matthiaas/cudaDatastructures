#include <math.h>

#include <cub/cub.cuh>
#include <iostream>

#include "../../cuda_utils.cuh"
#include "atomiccas.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARPSIZE 32
#define BLOCKSIZE 1024

namespace atomiccas {

template <typename T>
__global__ void add_as_accumuluated_requests(T *v, uint32_t iters) {
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  __shared__ T requests[warp_count * WARPSIZE];

  while (iters--) {
    int value = 1;
    int pos;

    WarpScan(temp_storage[warp_id]).ExclusiveSum(value, pos);

    requests[warp_id * WARPSIZE + pos] = value;
    __syncwarp();
    if (threadIdx.x % WARPSIZE == 0) {
      T sum = 0;
      for (int i = 0; i < WARPSIZE; i++) {
        sum += requests[warp_id * WARPSIZE + i];
      }

      while (1) {
        T val = *v;
        if (atomicCAS(v, val, val + sum) == val) {
          break;
        }
      }
    }
  }
}

template <typename T>
__global__ void add_as_requests(T *v, uint32_t iters) {
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  __shared__ T requests[warp_count * WARPSIZE];

  while (iters--) {
    int value = 1;
    int pos;

    WarpScan(temp_storage[warp_id]).ExclusiveSum(value, pos);

    requests[warp_id * WARPSIZE + pos] = value;
    __syncwarp();
    if (threadIdx.x % WARPSIZE == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
        while (1) {
          T val = *v;
          if (atomicCAS(v, val, val + requests[warp_id * WARPSIZE + i]) ==
              val) {
            break;
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void add_warp_inc(T *v, uint32_t iters) {
  while (iters--) {
    // https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
    auto active = cooperative_groups::coalesced_threads();
    T warp_res;
    if (active.thread_rank() == 0) {
      while (1) {
        int val = *v;
        if (atomicCAS(v, val, val + 1) == val) {
          warp_res = val;
          break;
        }
      }
    }
    active.shfl(warp_res, 0);
  }
}

template <typename T>
__global__ void add_trival(T *v, uint32_t iters) {
  while (iters--) {
    while (1) {
      int val = *v;
      if (atomicCAS(v, val, val + 1) == val) {
        break;
      }
    }
  }
}

template __global__ void add_as_accumuluated_requests(uint32_t *v,
                                                      uint32_t iters);
template __global__ void add_as_requests(uint32_t *v, uint32_t iters);
template __global__ void add_warp_inc(uint32_t *v, uint32_t iters);
template __global__ void add_trival(uint32_t *v, uint32_t iters);


template __global__ void add_as_accumuluated_requests(uint64_t *v,
                                                      uint32_t iters);
template __global__ void add_as_requests(uint64_t *v, uint32_t iters);
template __global__ void add_warp_inc(uint64_t *v, uint32_t iters);
template __global__ void add_trival(uint64_t *v, uint32_t iters);

}  // namespace atomiccas

// int main(void)
// {
//   int *v;
//   cudaMalloc(&v, sizeof(int));

//   int blocks = 1024 * 4;
//   int threads = BLOCKSIZE;

//   auto map = std::map<std::string, std::function<void(void)>>{
//     {"cadd_as_accumuluated_requests", [&] {
//     add_as_accumuluated_requests<<<blocks, threads>>>(v); }},
//     {"badd_as_requests", [&] { add_as_requests<<<blocks, threads>>>(v); }},
//     {"aadd_trival", [&] { add_trival<<<blocks, threads>>>(v); }},
//   };

//   auto init = [&] { cudaMemset(v, 0, sizeof(int)); };

//   auto validate = [&] {
//     int h_v;
//     cudaMemcpy(&h_v, v, sizeof(int), cudaMemcpyDeviceToHost);
//     return (h_v == (blocks * threads));
//   };

//   timeKernels(init, map, validate);

//   // Free memory
//   cudaFree(v);

//   return 0;
// }