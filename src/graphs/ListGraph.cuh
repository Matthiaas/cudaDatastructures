#ifndef LIST_GRAPH_H
#define LIST_GRAPH_H

#include <cstdint>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "graph_generation/default_graph.h"

struct ListGraph {
  uint32_t num_vertices;
  uint32_t num_edges;
  uint32_t *offsets;
  uint32_t *neighbors;
  bool on_device;

  ListGraph(uint32_t num_vertices, uint32_t num_edges, bool on_device) {
    this->num_vertices = num_vertices;
    this->num_edges = num_edges;
    this->on_device = on_device;
    if (on_device) {
      cudaMalloc(&offsets, (num_vertices + 1)* sizeof(uint32_t) );
      cudaMalloc(&neighbors, num_edges * sizeof(uint32_t));
    } else {
      offsets = static_cast<uint32_t *>(malloc((num_vertices + 1) * sizeof(uint32_t)));
      neighbors = static_cast<uint32_t *>(malloc(num_edges * sizeof(uint32_t)));
    }
  }

  void Free() {
    if (on_device) {
      cudaFree(offsets);
      cudaFree(neighbors);
    } else {
      free(offsets);
      free(neighbors);
    }
  }

  ListGraph CopyToDevice();
};

ListGraph ListGraph::CopyToDevice() {
  ListGraph res(this->num_vertices, this->num_edges, true);
  cudaMemcpy(res.offsets, offsets, sizeof(uint32_t) * (num_vertices + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(res.neighbors, neighbors, sizeof(uint32_t) * num_edges,
             cudaMemcpyHostToDevice);
  return res;
}

ListGraph ListGraphFromDefaultGraph(const DefaultGraph &graph) {
  ListGraph res(graph.NumNodes(), graph.NumEdges(), false);
  uint32_t edge_index = 0;
  const auto& edge_lists = graph.GetEdgeLists();
  for (uint32_t src = 0; src < edge_lists.size(); src++) {
    res.offsets[src] = edge_index;
    for (uint32_t dst : edge_lists[src]) {
      res.neighbors[edge_index] = dst;
      edge_index++;
    }
  }
  res.offsets[res.num_vertices] = edge_index;
  return res;
}


#endif
