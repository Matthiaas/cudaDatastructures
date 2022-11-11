
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cooperative_groups.h>

#include "Graph.h"

__host__ __device__ inline uint32_t UF_find(uint32_t *parents, uint32_t x) {
  while (parents[x] != x) {
    x = parents[x];
  }
  return x;
}

__device__ inline bool UF_union(uint32_t *parents, uint32_t src, uint32_t dst) {
  while (1) {
    uint32_t src_root = UF_find(parents, src);
    uint32_t dst_root = UF_find(parents, dst);
    if (src_root == dst_root) {
      return false;
    }

    if (atomicCAS(&parents[src_root], src_root, dst_root) == src_root) {
      return true;
    }
  }
  
}

__global__ void UF_find_kernel(uint32_t *parents, COOGraph graph) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  UF_union(parents, graph.srcs[idx], graph.dsts[idx]);
}