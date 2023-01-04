#ifndef CPU_BFS_H
#define CPU_BFS_H

#include <cinttypes>
#include <atomic>
#include <type_traits>
#include <optional>
#include <cstdlib>

#include "ListGraph.cuh"

struct MiniQueue {
  uint32_t* queue;
  uint32_t reader;
  uint32_t writer;
  uint32_t size;

  MiniQueue(size_t size) {
    queue = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
    this->size = size;
    reset();
  }
  ~MiniQueue() {
    free(queue);
  }

  void insert(uint32_t element) {
    auto pos = writer;
    writer = (writer + 1) % size;
    queue[pos] = element;
  }

  std::optional<uint32_t> get() {
    if (reader == writer) {
      return std::nullopt;
    }
    auto pos = reader;
    reader = (reader + 1) % size;
    return queue[pos];
  }

  void reset() {
    reader = 0;
    writer = 0;
  }
};


void RunCpuBfs(ListGraph& list_graph, uint32_t* distance, uint32_t start_node) {
  auto num_nodes = list_graph.num_vertices;
  auto offsets = list_graph.offsets;
  auto neighbors = list_graph.neighbors;

  auto queue = MiniQueue(num_nodes);
  for (auto i = 0; i < num_nodes; i++) {
    distance[i] = std::numeric_limits<uint32_t>::max();
  }

  queue.insert(start_node);
  distance[start_node] = 0;

  while (true) {
    auto node = queue.get();
    if (!node.has_value()) {
      break;
    }
    auto node_id = node.value();
    auto cur_distance = distance[node_id];
    auto start_edge = offsets[node_id];
    auto end_edge = offsets[node_id + 1];
    for (auto edge = start_edge; edge < end_edge; edge++) {
      auto neighbor = neighbors[edge];
      if (distance[neighbor] == std::numeric_limits<uint32_t>::max()) {
        distance[neighbor] = cur_distance + 1;
        queue.insert(neighbor);
      }
    }
  }
}




#endif // CPU_BFS_H
