#ifndef BFS_H
#define BFS_H

#include "Graph.cuh"

__global__ void bfs_kernel_start(COOGraph graph, uint32_t* distances,
                                 uint32_t start_node) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  if (thread_id == 0) {
    distances[start_node] = 0;
  }
}

__global__ void bfs_kernel_step(COOGraph graph, uint32_t* distances,
                                uint32_t start_node, bool* changed) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  for (auto i = thread_id; i < graph.num_edges; i += num_threads) {
    if (i >= graph.num_edges) {
      break;
    }
    const auto src = graph.srcs[i];
    const auto dst = graph.dsts[i];
    if (distances[src] != UINT32_MAX && distances[dst] == UINT32_MAX) {
      distances[dst] = distances[src] + 1;
      *changed = true;
    }
  }
}

// This only works for graphs with a single connected component
// This does not work, blocks need to work indeoendet of each other, so this
// could deadlock.
// __global__ void bfs_kernel_full(COOGraph graph, uint32_t* distances, uint32_t
// start_node) {
//     const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     const auto num_threads = blockDim.x * gridDim.x;

//     for (auto i = thread_id; i < graph.num_edges; i += num_threads) {
//         if (i >= graph.num_edges) {
//             break;
//         }
//         const auto src = graph.srcs[i];
//         const auto dst = graph.dsts[i];
//         if (distances[src] != UINT32_MAX  && distances[src] + 1 <
//         distances[dst]) {
//             distances[dst] = distances[src] + 1;
//         }
//     }
// }

void bfs(COOGraph graph, uint32_t* distances, uint32_t start_node) {
  const auto num_threads = 256;
  const auto num_blocks = 256;

  bfs_kernel_start<<<1, 1>>>(graph, distances, start_node);
  cudaDeviceSynchronize();

  bool changed = true;
  bool* changed_d;
  cudaMalloc(&changed_d, sizeof(bool));

  while (changed) {
    changed = false;
    cudaMemcpy(changed_d, &changed, sizeof(bool), cudaMemcpyHostToDevice);
    bfs_kernel_step<<<num_blocks, num_threads>>>(graph, distances, start_node,
                                                 changed_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&changed, changed_d, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  cudaFree(changed_d);
}

#endif  // BFS_H
