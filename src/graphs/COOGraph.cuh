#ifndef COO_GRAPH_H
#define COO_GRAPH_H

#include <cstdint>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "graph_generation/default_graph.h"

#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                    << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#endif

struct COOGraph {
  uint32_t *srcs;
  uint32_t *dsts;
  uint32_t num_vertices;
  uint32_t num_edges;
  
  bool on_device;

  COOGraph() {
    num_vertices = 0;
    num_edges = 0;
    on_device = false;
    srcs = nullptr;
    dsts = nullptr;
  }

  COOGraph(uint32_t num_vertices, uint32_t num_edges, bool on_device) {
    this->num_vertices = num_vertices;
    this->num_edges = num_edges;
    this->on_device = on_device;
    
    if (on_device) {
      cudaMalloc(&srcs, num_edges * sizeof(uint32_t));CUERR
      cudaMalloc(&dsts, num_edges * sizeof(uint32_t));CUERR
    } else {
      srcs = static_cast<uint32_t *>(malloc(num_edges * sizeof(uint32_t)));
      dsts = static_cast<uint32_t *>(malloc(num_edges * sizeof(uint32_t)));
    }
  }

  void Free() {
    if (on_device) {
      cudaFree(srcs);
      cudaFree(dsts);
    } else {
      free(srcs);
      free(dsts);
    }
  }

  COOGraph CopyToDevice();
};

COOGraph COOGraph::CopyToDevice() {
  COOGraph res(this->num_vertices, this->num_edges, true);
  cudaMemcpy(res.srcs, srcs, sizeof(uint32_t) * num_edges,
             cudaMemcpyHostToDevice);CUERR
  cudaMemcpy(res.dsts, dsts, sizeof(uint32_t) * num_edges,
             cudaMemcpyHostToDevice);CUERR
  return res;
}

COOGraph COOFromDefaultGraph(const DefaultGraph &graph) {
  COOGraph res(graph.NumNodes(), graph.NumEdges(), false);
  uint32_t edge_index = 0;
  const auto &edge_lists = graph.GetEdgeLists();
  for (uint32_t src = 0; src < edge_lists.size(); src++) {
    for (uint32_t dst : edge_lists[src]) {
      res.srcs[edge_index] = src;
      res.dsts[edge_index] = dst;
      edge_index++;
    }
  }
  return res;
}

#endif
