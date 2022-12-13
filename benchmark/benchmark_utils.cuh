#ifndef BECHMARK_UTILS_H
#define BECHMARK_UTILS_H

#include <math.h>

#include <chrono>
#include <cub/cub.cuh>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace benchmark {

inline float timeKernel(std::function<void(void)> invoke_kernel) {
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
inline void runGpuBenchmark(T* bechmark, size_t num_iterations, size_t blocks,
                     size_t threads, size_t ring_size) {
  // std::cout << "Running GPU bechmark for " << bechmark->name() << std::endl;
  // std::cout << "----------------------------------------" << std::endl;

  bechmark->GpuInit();
  float ms_time = timeKernel(
      std::bind(&T::GpuRun, bechmark, num_iterations, blocks, threads));
  bechmark->GpuCleanup();
  std::cout << "GPU," << bechmark->name() << "," << threads << "," << blocks
            << "," << num_iterations << "," << ring_size << "," << ms_time << std::endl;
  // std::cout << "Time: " << ms_time << " ms" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
}

template <typename T>
inline void runCpuBenchmark(T* bechmark, size_t num_iterations, size_t threads, size_t ring_size) {
  // std::cout << "Running CPU bechmark for " << bechmark->name() << std::endl;
  // std::cout << "----------------------------------------" << std::endl;

  bechmark->CpuInit();
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  bechmark->CpuRun(num_iterations, threads);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  bechmark->CpuCleanup();
  double time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count() /
      1000.0;
  std::cout << "CPU," << bechmark->name() << "," << threads << "," << 1 << ","
            << num_iterations << "," << ring_size  << "," << time << std::endl;
  // std::cout << "Time: " <<   << " ms" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
  // std::cout << "----------------------------------------" << std::endl;
}

inline void timeKernels(
    std::function<void(void)> initstate,
    const std::map<std::string, std::function<void(void)>>& kernels,
    std::function<bool(void)> validate) {
  for (auto it = kernels.begin(); it != kernels.end(); it++) {
    initstate();
    float time = timeKernel(it->second);
    bool validated = validate();
    std::cout << it->first << "," << validated << "," << time << std::endl;
  }
}

}  // namespace benchmark

#endif  // BECHMARK_UTILS_H