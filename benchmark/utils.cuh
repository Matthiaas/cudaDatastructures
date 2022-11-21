#ifndef BECHMARK_UTILS_H
#define BECHMARK_UTILS_H

#include <chrono>
#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

namespace benchmark {

float timeKernel(std::function<void(void)> invoke_kernel)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  invoke_kernel();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}

template <typename T>
void runGpuBenchmark(T* bechmark, size_t num_iterations, size_t blocks, size_t threads) {  
  // std::cout << "Running GPU bechmark for " << bechmark->name() << std::endl;
  // std::cout << "----------------------------------------" << std::endl;

  bechmark->GpuInit();
  float ms_time = timeKernel(std::bind(&T::GpuRun, bechmark, num_iterations, blocks, threads)); 
  bechmark->GpuCleanup();
  std::cout << "GPU," << bechmark->name()  << "," << threads << "," << blocks << "," << num_iterations << "," << ms_time << std::endl;
  // std::cout << "Time: " << ms_time << " ms" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
}

template <typename T>
void runCpuBenchmark(T* bechmark, size_t num_iterations, size_t threads) {  
  // std::cout << "Running CPU bechmark for " << bechmark->name() << std::endl;
  // std::cout << "----------------------------------------" << std::endl;

  bechmark->CpuInit();
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bechmark->CpuRun(num_iterations, threads);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  bechmark->CpuCleanup();
  double time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
  std::cout << "CPU," << bechmark->name()  << "," << threads << "," << 1 << "," << num_iterations << "," << time << std::endl;
  // std::cout << "Time: " <<   << " ms" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
}

void timeKernels(
  std::function<void(void)> initstate,
  const std::map<std::string, std::function<void(void)>>& kernels, 
  std::function<bool(void)> validate)
{
  for (auto it = kernels.begin(); it != kernels.end(); it++) {
    initstate();
    float time = timeKernel(it->second);
    bool validated = validate();
    std::cout << it->first << "," << validated << "," << time << std::endl;
  }
}

}

#endif // BECHMARK_UTILS_H