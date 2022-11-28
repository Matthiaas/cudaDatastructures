#ifndef BFS_H
#define BFS_H

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

void bfs(COOGraph graph, uint32_t* distances, uint32_t start_node) {
  const auto num_threads = 512;
  const auto num_blocks = 1024;

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

void bfs_iterations_based(COOGraph graph, uint32_t* distances, uint32_t start_node) {
  const auto num_threads = 512;
  const auto num_blocks = 1024;

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

template <size_t group_size = 1>
__global__ void bfs_kernel_step(ListGraph graph, uint32_t* distances,
                                SimpleQueue queue_in, SimpleQueue queue_out) {
  namespace cg = cooperative_groups;

  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  auto queue_in_size = *queue_in.size;

  const auto group = cg::tiled_partition<group_size>(cg::this_thread_block());
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
        auto res = atomicCAS(&distances[dst], UINT32_MAX, distances[node] + 1);
        if (res == UINT32_MAX) {
          const auto index = atomicAdd(queue_out.size, 1);
          queue_out.data[index] = dst;
        }
      }
    }
  }
}

template <size_t group_size = 1>
__global__ void bfs_kernel_step_sharework(ListGraph graph, uint32_t* distances,
                                SimpleQueue queue_in, SimpleQueue queue_out,
                                uint32_t* queue_pos) {
  namespace cg = cooperative_groups;
  auto queue_in_size = *queue_in.size;
  const auto group = cg::tiled_partition<group_size>(cg::this_thread_block());

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
        auto res = atomicCAS(&distances[dst], UINT32_MAX, distances[node] + 1);
        if (res == UINT32_MAX) {
          const auto index = atomicAdd(queue_out.size, 1);
          queue_out.data[index] = dst;
        }
      }
    }
  }
}

template <size_t group_size = 1>
void bfs(ListGraph graph, uint32_t* distances, uint32_t start_node) {
  const auto num_threads = 512;
  const auto num_blocks = 256;

  SimpleQueue queue_in;
  SimpleQueue queue_out;

  cudaMalloc(&queue_in.data, graph.num_vertices * sizeof(uint32_t));
  cudaMalloc(&queue_in.size, sizeof(uint32_t));
  cudaMalloc(&queue_out.data, graph.num_vertices * sizeof(uint32_t));
  cudaMalloc(&queue_out.size, sizeof(uint32_t));

  const auto block_count_for_init =
      (graph.num_vertices + num_threads - 1) / num_threads;

  bfs_kernel_start<<<block_count_for_init, num_threads>>>(graph, distances,
                                                          start_node, queue_in);
  // Not sure If I need this is needed.
  // cudaDeviceSynchronize();
  uint32_t queue_out_size = 1;

  while (queue_out_size) {
    cudaMemset(queue_out.size, 0, sizeof(uint32_t));
    bfs_kernel_step<group_size><<<num_blocks, num_threads>>>(graph, distances, queue_in,
                                                 queue_out);
    cudaMemcpy(&queue_out_size, queue_out.size, sizeof(uint32_t),
               cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
    std::swap(queue_in, queue_out);
  }

  cudaFree(queue_in.data);
  cudaFree(queue_in.size);
  cudaFree(queue_out.data);
  cudaFree(queue_out.size);
}


template <size_t group_size = 1>
void bfs_sharework(ListGraph graph, uint32_t* distances, uint32_t start_node) {
  const auto num_threads = 512;
  const auto num_blocks = 256;

  SimpleQueue queue_in;
  SimpleQueue queue_out;

  uint32_t *queue_pos;
  cudaMalloc(&queue_pos, sizeof(uint32_t));

  cudaMalloc(&queue_in.data, graph.num_vertices * sizeof(uint32_t));
  cudaMalloc(&queue_in.size, sizeof(uint32_t));
  cudaMalloc(&queue_out.data, graph.num_vertices * sizeof(uint32_t));
  cudaMalloc(&queue_out.size, sizeof(uint32_t));

  const auto block_count_for_init =
      (graph.num_vertices + num_threads - 1) / num_threads;

  bfs_kernel_start<<<block_count_for_init, num_threads>>>(graph, distances,
                                                          start_node, queue_in);
  // Not sure If I need this is needed.
  // cudaDeviceSynchronize();
  uint32_t queue_out_size = 1;

  while (queue_out_size) {
    cudaMemset(queue_out.size, 0, sizeof(uint32_t));
    cudaMemset(queue_pos, 0, sizeof(uint32_t));
    bfs_kernel_step_sharework<group_size><<<num_blocks, num_threads>>>(graph, distances, queue_in,
                                                 queue_out, queue_pos);
    cudaMemcpy(&queue_out_size, queue_out.size, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    std::swap(queue_in, queue_out);
  }

  cudaFree(queue_pos);
  cudaFree(queue_in.data);
  cudaFree(queue_in.size);
  cudaFree(queue_out.data);
  cudaFree(queue_out.size);

}

}

#endif  // BFS_H
