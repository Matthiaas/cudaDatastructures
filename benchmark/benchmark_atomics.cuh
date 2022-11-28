#ifndef BENCHMARK_ATOMICS_H_
#define BENCHMARK_ATOMICS_H_

#include "atomicscontention/atomicadd.cuh"
#include "atomicscontention/atomiccas.cuh"
#include "benchmark_params.h"
#include "benchmark_utils.cuh"

#include <functional>
#include <iostream>

template<typename T>
using AtomicsMap = std::map<std::string, std::function<void(size_t, size_t, size_t, T*)>>;

template<typename T>
AtomicsMap<T> atomicadd_map = {
  {"add_as_accumuluated_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v) { 
    atomicadd::add_as_accumuluated_requests<<<blocks, threads>>>(v,iters); 
  }},
  {"add_as_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
    atomicadd::add_as_requests<<<blocks, threads>>>(v,iters); 
  }},
  {"add_trival", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
    atomicadd::add_trival<<<blocks, threads>>>(v,iters); 
  }},
};

template<typename T>
AtomicsMap<T> atomiccas_map = {
  {"add_as_accumuluated_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v) { 
    atomiccas::add_as_accumuluated_requests<<<blocks, threads>>>(v,iters); 
  }},
  {"add_as_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
    atomiccas::add_as_requests<<<blocks, threads>>>(v,iters); 
  }},
  {"add_trival", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
    atomiccas::add_trival<<<blocks, threads>>>(v,iters); 
  }},
};

// This could be so easy if we could use C++20 bind front
template <typename T>
void runAtomicBenchMark(AtomicsMap<T> map,
                        const benchmark::BenchParams& params) {
  T* v;
  cudaMalloc(&v, sizeof(int));
  auto init = [&] { cudaMemset(v, 0, sizeof(int)); };

  auto validate = [&] {
    int h_v;
    cudaMemcpy(&h_v, v, sizeof(int), cudaMemcpyDeviceToHost);
    return (h_v ==
            (params.gpu.blocks * params.gpu.threads * params.gpu.iterations));
  };

  for (auto [name, func] : map) {
    init();
    float time = benchmark::timeKernel(std::bind(
        func, params.gpu.threads, params.gpu.blocks, params.gpu.iterations, v));
    bool validated = validate();
    std::cout << name << "," << params.gpu.blocks << "," << params.gpu.threads
              << "," << params.gpu.iterations << "," << (8 * sizeof(T)) << ","
              << validated << "," << time << std::endl;
  }

  cudaFree(v);
}


void runAtomicsBenchMark(const benchmark::BenchParams& params) {
   if (params.atomicadd) {
    bool found = false;
    if (params.data_type.find("64") != std::string::npos) {
      runAtomicBenchMark(atomicadd_map<uint64_t>, params);
      found = true;
    }
    if (params.data_type.find("32") != std::string::npos) {
      runAtomicBenchMark(atomicadd_map<uint32_t>, params);
      found = true;
    }
    if (!found) {
      std::cout << "Invalid data type for atomicadd: " << params.data_type
                << std::endl;
    }
  }

  if (params.atomiccas) {
    bool found = false;
    if (params.data_type.find("64") != std::string::npos) {
      runAtomicBenchMark(atomiccas_map<uint64_t>, params);
      found = true;
    }
    if (params.data_type.find("32") != std::string::npos) {
      runAtomicBenchMark(atomiccas_map<uint32_t>, params);
      found = true;
    }
    if (!found) {
      std::cout << "Invalid data type for atomicadd: " << params.data_type
                << std::endl;
    }
  }
}


#endif  // BENCHMARK_ATOMICS_H_
