#ifndef BENCHMARK_CACHING_H
#define BENCHMARK_CACHING_H

#include "caching/caching.cuh"
#include "benchmark_params.h"
#include "benchmark_utils.cuh"

using CachinMap = std::map<std::string, std::function<void(size_t, size_t, uint32_t*, size_t)>>;

CachinMap caching_map = {
  {"ReadAndWriteSameSlot", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteSameSlot<<<blocks, threads>>>(data,data_size,false); 
  }},
  {"ReadAndWriteDifferentSlot", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteDifferentSlot<<<blocks, threads>>>(data,data_size,false); 
  }},
  {"ReadAndWriteDifferentCacheLine", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteDifferentCacheLine<<<blocks, threads>>>(data,data_size,false); 
  }},
  {"ReadAndOneWriteSameSlot", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteSameSlot<<<blocks, threads>>>(data,data_size,true); 
  }},
  {"ReadAndOneWriteDifferentSlot", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteDifferentSlot<<<blocks, threads>>>(data,data_size,true); 
  }},
  {"ReadAndOneWriteDifferentCacheLine", [=] (size_t threads, size_t blocks, uint32_t* data, size_t data_size) { 
    caching::ReadAndWriteDifferentCacheLine<<<blocks, threads>>>(data,data_size,true); 
  }},
};

void runCachingBenchMark(CachinMap map, const benchmark::BenchParams& params) {
  uint32_t* v;
  cudaMalloc(&v, sizeof(uint32_t) * params.gpu.iterations);
  cudaMemset(v, 0, sizeof(uint32_t) * params.gpu.iterations);

  for (auto [name, func] : map) {
    float time = benchmark::timeKernel(std::bind(
        func, params.gpu.threads, params.gpu.blocks, v, params.gpu.iterations));
    std::cout << name << "," << params.gpu.blocks << "," << params.gpu.threads
              << "," << params.gpu.iterations << ","
              << "," << time << std::endl;
  }

  cudaFree(v);
}

#endif  // BENCHMARK_CACHING_H
