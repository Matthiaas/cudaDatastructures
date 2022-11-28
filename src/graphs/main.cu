#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.h"
#include "UnionFind.h"

// Kernel function to add the elements of two arrays

int main(void)
{
  // COOGraph graph(2048,4096, false);
  // graph.SetEdge(0, 0, 1);
  // graph.SetEdge(1, 1, 2);
  // graph.SetEdge(2, 3, 4);
  // graph.SetEdge(3, 5, 6);
  // graph.SetEdge(4, 7, 3);
  // graph.SetEdge(5, 7, 5);
  // graph.SetEdge(6, 7, 2);


  // COOGraph dev_graph = graph.CopyToDevice();

  // uint32_t *parents;
  // cudaMalloc((void**)&parents, sizeof(uint32_t) * graph.num_vertices);

  // uint32_t *host_parents = new uint32_t[graph.num_vertices];
  // for (uint32_t i = 0; i < graph.num_vertices; i++) {
  //   host_parents[i] = i;
  // }
  // cudaMemcpy(parents, host_parents, sizeof(uint32_t) * graph.num_vertices, cudaMemcpyHostToDevice);


  // UF_find_kernel<<<4,512>>>(parents, dev_graph);

  // // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();

  
  // // copy from gpu
  // cudaMemcpy(host_parents, parents, sizeof(uint32_t) * graph.num_vertices, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 10; i++)
  // {
  //   std::cout << i << " " << host_parents[i] << " " << UF_find(host_parents, i) << std::endl;
  // }


  // return 0;
}