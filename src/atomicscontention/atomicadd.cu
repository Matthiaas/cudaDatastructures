#include "atomicadd.cuh"

#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../../cuda_utils.cuh"

#define WARPSIZE 32
#define BLOCKSIZE 1024 

namespace atomicadd {

template <typename T>
__global__
void add_as_accumuluated_requests(T *v, uint32_t iters)
{
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  while (iters--) {
    int value = 1;
    int prefix_sum;

    WarpScan(temp_storage[warp_id]).ExclusiveSum(value, prefix_sum);

    __syncwarp();
    if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
      atomicAdd(v, prefix_sum);
    }
  }
  
}

template <typename T>
__global__
void add_as_requests(T *v, uint32_t iters)
{
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  __shared__ int requests[warp_count * WARPSIZE];

  while (iters--) {
    int value = 1;
    int pos;

    WarpScan(temp_storage[warp_id]).ExclusiveSum(value, pos);

    requests[warp_id * WARPSIZE + pos] = value;
    __syncwarp();
    if(threadIdx.x % WARPSIZE == 0) {
      for (int i = 0; i < WARPSIZE; i++) {
        atomicAdd(v, requests[warp_id * WARPSIZE + i]);
      }
    }
  }
  

  
  
}

template <typename T>
__global__
void add_trival(T *v, uint32_t iters)
{
  while (iters--) {
    atomicAdd(v, 1);
  }
}

template __global__ void add_as_accumuluated_requests(uint32_t *v, uint32_t iters);
template __global__ void add_as_requests(uint32_t *v, uint32_t iters);
template __global__ void add_trival(uint32_t *v, uint32_t iters);

template __global__ void add_as_accumuluated_requests(uint64_t *v, uint32_t iters);
template __global__ void add_as_requests(uint64_t *v, uint32_t iters);
template __global__ void add_trival(uint64_t *v, uint32_t iters);

}



// int main(void)
// {
//   uint32_t *v;
//   cudaMalloc(&v, sizeof(int));


  
//   int blocks = 1024;
//   int threads = BLOCKSIZE;
//   int iters = 1000;

//   auto map = std::map<std::string, std::function<void(void)>>{
//     {"add_as_accumuluated_requests", [=] { atomicadd::add_as_accumuluated_requests<<<blocks, threads>>>(v,iters); }},
//     {"add_as_requests", [=] { atomicadd::add_as_requests<<<blocks, threads>>>(v,iters); }},
//     {"add_trival", [=] { atomicadd::add_trival<<<blocks, threads>>>(v,iters); }},
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