#ifndef BFS_H
#define BFS_H

#define _CG_ABI_EXPERIMENTAL
#define _CG_CPP11_FEATURES
#include <cooperative_groups.h>

#include "COOGraph.cuh"
#include "ListGraph.cuh"


namespace coo_graph_aglos {

__global__ void bfs_kernel_start(COOGraph graph, uint32_t* distances,
                                 uint32_t start_node) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < graph.num_vertices) {
    distances[thread_id] = (thread_id == start_node) ? 0 : UINT32_MAX;
  }
}

__global__ void bfs_kernel_step_iteration_based(COOGraph graph, uint32_t* distances,
                                uint32_t start_node, uint32_t* changed,
                                uint32_t iteration) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  if (thread_id == 0) {
  }

  for (auto i = thread_id; i < graph.num_edges; i += num_threads) {
    if (i >= graph.num_edges) {
      break;
    }
    const auto src = graph.srcs[i];
    const auto dst = graph.dsts[i];
    if (distances[src] == iteration && distances[dst] == UINT32_MAX) {
      distances[dst] = distances[src] + 1;
      *changed = 1;
    }
  }
}

__global__ void bfs_kernel_step(COOGraph graph, uint32_t* distances,
                                uint32_t start_node, uint32_t* changed) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  if (thread_id == 0) {
  }

  for (auto i = thread_id; i < graph.num_edges; i += num_threads) {
    if (i >= graph.num_edges) {
      break;
    }
    const auto src = graph.srcs[i];
    const auto dst = graph.dsts[i];
    if (distances[src] != UINT32_MAX && distances[src] + 1 < distances[dst]) {
      atomicMin(&distances[dst], distances[src] + 1);
      *changed = 1;
    }
  }
}

void bfs(COOGraph graph, uint32_t* distances, uint32_t start_node,
         uint32_t num_blocks, uint32_t num_threads) {
  const auto block_count_for_init =
      (graph.num_vertices + num_threads - 1) / num_threads;

  bfs_kernel_start<<<block_count_for_init, num_threads>>>(graph, distances,
                                                          start_node);
  CUERR
  cudaDeviceSynchronize();

  uint32_t changed = true;
  uint32_t* changed_d;
  cudaMalloc(&changed_d, sizeof(uint32_t));

  while (changed) {
    cudaMemset(changed_d, 0, sizeof(uint32_t));
    CUERR
    bfs_kernel_step<<<num_blocks, num_threads>>>(graph, distances, start_node,
                                                 changed_d);
    CUERR
    cudaMemcpy(&changed, changed_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }

  cudaFree(changed_d);
}

void bfs_iterations_based(COOGraph graph, uint32_t* distances,
                          uint32_t start_node, uint32_t num_blocks,
                          uint32_t num_threads) {

  const auto block_count_for_init =
      (graph.num_vertices + num_threads - 1) / num_threads;

  bfs_kernel_start<<<block_count_for_init, num_threads>>>(graph, distances,
                                                          start_node);
  CUERR
  cudaDeviceSynchronize();

  uint32_t changed = true;
  uint32_t* changed_d;
  cudaMalloc(&changed_d, sizeof(uint32_t));

  for (uint32_t iteration = 0; changed; ++iteration) {
    cudaMemset(changed_d, 0, sizeof(uint32_t));
    CUERR
    bfs_kernel_step_iteration_based<<<num_blocks, num_threads>>>(graph, distances, start_node,
                                                 changed_d, iteration);
    CUERR
    cudaMemcpy(&changed, changed_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }

  cudaFree(changed_d);
}
}

namespace list_graph_aglos {

struct SimpleQueue {
  uint32_t* data;
  uint32_t* size;
};

__global__ void bfs_kernel_start(ListGraph graph, uint32_t* distances,
                                 uint32_t start_node, SimpleQueue queue) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id == 0) {
    queue.data[0] = start_node;
    *queue.size = 1;
  }

  if (thread_id < graph.num_vertices) {
    distances[thread_id] = (thread_id == start_node) ? 0 : UINT32_MAX;
  }
}

struct Empty {};

template <typename T>
__device__ __forceinline__ cooperative_groups::thread_block GetCurrentThreadBlock(T& t) {
  if constexpr (std::is_same<Empty,T>::value) {
    return cooperative_groups::this_thread_block();
  } else {
    return cooperative_groups::experimental::this_thread_block(t);
  }
}


template <size_t group_size = 1, bool collect = true>
__global__ void bfs_kernel_step(ListGraph graph, uint32_t* distances,
                                SimpleQueue queue_in, SimpleQueue queue_out) {
  namespace cg = cooperative_groups;

  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  auto queue_in_size = *queue_in.size;

  static constexpr bool use_experimental = (group_size > 32);
  // Only put additional memory on stack when needed.
  __shared__ std::conditional_t<
      use_experimental, cg::experimental::block_tile_memory<8, 1024>, Empty>
      shared;
  cg::thread_block thb = GetCurrentThreadBlock(shared);
  const auto group =
      cooperative_groups::experimental::tiled_partition<group_size>(thb);

  const auto num_groups = num_threads / group_size;
  const size_t group_id = thread_id / group_size;

  for (auto i = group_id; i < queue_in_size; i += num_groups) {
    if (i >= queue_in_size) {
      break;
    }
    const uint32_t node = queue_in.data[i];

    for (auto j = graph.offsets[node] + group.thread_rank();
         j < graph.offsets[node + 1]; j += group_size) {
      const auto dst = graph.neighbors[j];
      if (distances[dst] == UINT32_MAX) {
        if constexpr (collect) {
          auto res = atomicCAS(&distances[dst], UINT32_MAX, distances[node] + 1);
          if (res == UINT32_MAX) {
            const auto index = atomicAdd(queue_out.size, 1);
            queue_out.data[index] = dst;
          }
        } else {
          // This is save since if it get updates by multiple threads they 
          // all wirte the same result.
          distances[dst] = distances[node] + 1;
        }
      }
    }
  }
}

template <size_t group_size = 1, bool collect = true>
__global__ void bfs_kernel_step_sharework(ListGraph graph, uint32_t* distances,
                                SimpleQueue queue_in, SimpleQueue queue_out,
                                uint32_t* queue_pos) {
  namespace cg = cooperative_groups;
  auto queue_in_size = *queue_in.size;
  
  static constexpr bool use_experimental = (group_size > 32);
  // Only put additional memory on stack when needed.
  __shared__ std::conditional_t<
      use_experimental, cg::experimental::block_tile_memory<8, 1024>, Empty>
      shared;
  cg::thread_block thb = GetCurrentThreadBlock(shared);
  const auto group =
      cooperative_groups::experimental::tiled_partition<group_size>(thb);

  while (true) {
    uint32_t i;
    if (group.thread_rank() == 0) {
      i = atomicAdd(queue_pos, 1);
    }
    i = group.shfl(i, 0);

    if (i >= queue_in_size) {
      break;
    }
    const uint32_t node = queue_in.data[i];
    for (auto j = graph.offsets[node] + group.thread_rank();
         j < graph.offsets[node + 1]; j += group_size) {
      const auto dst = graph.neighbors[j];
      if (distances[dst] == UINT32_MAX) {
        if constexpr (collect) {
          auto res = atomicCAS(&distances[dst], UINT32_MAX, distances[node] + 1);
          if (res == UINT32_MAX) {
            const auto index = atomicAdd(queue_out.size, 1);
            queue_out.data[index] = dst;
          }
        } else {
          // This is save since if it get updates by multiple threads they 
          // all wirte the same result.
          distances[dst] = distances[node] + 1;
        }
      }
    }
  }
}

__global__ void collectNodes(SimpleQueue queue, uint32_t* distances, uint32_t n,
                             uint32_t distance) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  for (uint32_t i = thread_id; ; i += num_threads) {
    if (i >= n) {
      break;
    }
    if(distances[i] == distance) {
      const auto index = atomicAdd(queue.size, 1);
      queue.data[index] = i;
    }
  }
}

template <size_t group_size = 1, bool share_work = false, bool collect_in_upate = true>
void bfs(ListGraph graph, uint32_t* distances, uint32_t start_node,
         uint32_t num_blocks, uint32_t num_threads) {
  SimpleQueue queue_in;
  SimpleQueue queue_out;
  uint32_t *queue_pos;
  

  cudaMalloc(&queue_in.data, graph.num_vertices * sizeof(uint32_t));
  cudaMalloc(&queue_in.size, sizeof(uint32_t));
  if constexpr (collect_in_upate) {
    cudaMalloc(&queue_out.data, graph.num_vertices * sizeof(uint32_t));
    cudaMalloc(&queue_out.size, sizeof(uint32_t));
    cudaMemset(queue_out.size, 0, sizeof(uint32_t));
  }
  if constexpr (share_work) {
    cudaMalloc(&queue_pos, sizeof(uint32_t));
  }

  const auto block_count_for_init =
      (graph.num_vertices + num_threads - 1) / num_threads;

  bfs_kernel_start<<<block_count_for_init, num_threads>>>(graph, distances,
                                                          start_node, queue_in);
  uint32_t queue_out_size = 1;
  
  for (uint32_t iteration = 1; queue_out_size; iteration++) {
    if constexpr (share_work) {
      cudaMemset(queue_pos, 0, sizeof(uint32_t));
      bfs_kernel_step_sharework<group_size, collect_in_upate>
          <<<num_blocks, num_threads>>>(graph, distances, queue_in, queue_out,
                                        queue_pos);
    } else {
      bfs_kernel_step<group_size, collect_in_upate>
          <<<num_blocks, num_threads>>>(graph, distances, queue_in, queue_out);
    }
    if constexpr (!collect_in_upate) {
      cudaMemset(queue_in.size, 0, sizeof(uint32_t));
      collectNodes<<<block_count_for_init, num_threads>>>(
          queue_in, distances, graph.num_vertices, iteration);
      cudaMemcpy(&queue_out_size, queue_in.size, sizeof(uint32_t),
                 cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(&queue_out_size, queue_out.size, sizeof(uint32_t),
               cudaMemcpyDeviceToHost); 
      std::swap(queue_in, queue_out);
      cudaMemset(queue_out.size, 0, sizeof(uint32_t));
    }
  }

  cudaFree(queue_in.data);
  cudaFree(queue_in.size);
  if constexpr (collect_in_upate) {
    cudaFree(queue_out.data);
    cudaFree(queue_out.size);
  }
  if constexpr (share_work) {
    cudaFree(queue_pos);
  }
}
}

#endif  // BFS_H
