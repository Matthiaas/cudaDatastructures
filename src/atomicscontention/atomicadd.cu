#include "atomicadd.cuh"

#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#define WARPSIZE 32
#define BLOCKSIZE 1024 

namespace atomicadd {

__global__
void add_as_accumuluated_requests(int *v)
{
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  int value = 1;
  int prefix_sum;

  WarpScan(temp_storage[warp_id]).ExclusiveSum(value, prefix_sum);

  __syncwarp();
  if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
    atomicAdd(v, prefix_sum);
  }
  
}


__global__
void add_as_requests(int *v)
{
  const int warp_count = BLOCKSIZE / WARPSIZE;
  int warp_id = threadIdx.x / WARPSIZE;

  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

  __shared__ int requests[warp_count * WARPSIZE];

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

__global__
void add_trival(int *v)
{
  atomicAdd(v, 1);
}

}

// int main(void)
// {
//   int *v;
//   cudaMalloc(&v, sizeof(int));


  
//   int blocks = 1024 * 4;
//   int threads = BLOCKSIZE;

//   auto map = std::map<std::string, std::function<void(void)>>{
//     {"add_as_accumuluated_requests", [=] { add_as_accumuluated_requests<<<blocks, threads>>>(v); }},
//     {"add_as_requests", [=] { add_as_requests<<<blocks, threads>>>(v); }},
//     {"add_trival", [=] { add_trival<<<blocks, threads>>>(v); }},
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