#ifndef GRAPH_H
#define GRAPH_H

#include <cstdint>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct COOGraph {
  uint32_t num_vertices;
  uint32_t num_edges;
  uint32_t *srcs;
  uint32_t *dsts;

  COOGraph(uint32_t num_vertices, uint32_t num_edges, bool on_device) {
    this->num_vertices = num_vertices;
    this->num_edges = num_edges;
    if (on_device) {
      cudaMalloc(&srcs, num_edges * sizeof(uint32_t));
      cudaMalloc(&dsts, num_edges * sizeof(uint32_t));
    } else {
      srcs = static_cast<uint32_t *>(malloc(num_edges * sizeof(uint32_t)));
      dsts = static_cast<uint32_t *>(malloc(num_edges * sizeof(uint32_t)));
    }
  }

  COOGraph CopyToDevice();
  void SetEdge(uint32_t edge_id, uint32_t src, uint32_t dst);
};

COOGraph COOGraph::CopyToDevice() {
  COOGraph res(this->num_vertices, this->num_edges, true);
  cudaMemcpy(res.srcs, srcs, sizeof(uint32_t) * num_edges,
             cudaMemcpyHostToDevice);
  cudaMemcpy(res.dsts, dsts, sizeof(uint32_t) * num_edges,
             cudaMemcpyHostToDevice);
  return res;
}

void COOGraph::SetEdge(uint32_t edge_id, uint32_t src, uint32_t dst) {
  srcs[edge_id] = src;
  dsts[edge_id] = dst;
}

#endif
