#ifndef BENCHMARK_COPYING_H
#define BENCHMARK_COPYING_H

#include "benchmark_params.h"
#include "benchmark_utils.cuh"

#include <iostream>

void runHashMapCopyingBenchMark( const benchmark::BenchParams& params) {
  
  static constexpr size_t max_count = 1ul << 26;
  using data_type = uint32_t;

  data_type* a;
  data_type* b;
  data_type* hosta;
  data_type* hostb;
  cudaMalloc(&a, sizeof(data_type) * max_count);
  cudaMalloc(&b, sizeof(data_type) * max_count);
  hosta = static_cast<data_type*>(malloc(sizeof(data_type) * max_count));
  hostb = static_cast<data_type*>(malloc(sizeof(data_type) * max_count));


  for (size_t i = 1; i <= max_count; i *= 2) {
    float ms_insert = benchmark::timeKernel([&] () {
      cudaMemcpyAsync(a, hosta, sizeof(data_type) * i,
             cudaMemcpyHostToDevice);
      cudaMemcpyAsync(b, hostb, sizeof(data_type) * i,
             cudaMemcpyHostToDevice);
    });

    float ms_find = benchmark::timeKernel([&] () {
      cudaMemcpyAsync(a, hosta, sizeof(data_type) * i,
             cudaMemcpyHostToDevice);
      cudaMemcpyAsync(hostb, b, sizeof(data_type) * i,
             cudaMemcpyDeviceToHost);
    });
    
    std::cout << i << ' ' << ms_insert << ' ' << ms_find << std::endl;
  }

 
}

#endif  // BENCHMARK_COPYING_H
