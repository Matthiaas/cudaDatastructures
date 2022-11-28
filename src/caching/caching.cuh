#ifndef CACHING_H
#define CACHING_H

#include <cinttypes>

namespace caching {

__device__ static uint32_t read(volatile uint32_t *addr) { return *addr; }

__global__ void ReadAndWriteSameSlot(uint32_t *data, size_t size,
                                     bool only_one) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1024) {
      uint32_t res = read(&data[i + idx]);
      // if (!only_one || idx == 0) {
      //   data[i] = res;
      // }
      __threadfence();
    }
  }
}

__global__ void ReadAndWriteDifferentSlot(uint32_t *data, size_t size,
                                          bool only_one) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1024) {
      uint32_t res = read(&data[i + idx]);
      // if (!only_one || idx == 0) {
      //   data[i + 8] = res;
      // }
      __threadfence();
    }
  }
}

__global__ void ReadAndWriteDifferentCacheLine(uint32_t *data, size_t size,
                                               bool only_one) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1024) {
      uint32_t res = read(&data[i + idx]);
      // if (!only_one || idx == 0) {
      //   data[i + 512] = res;
      // }
      __threadfence();
    }
  }
}

}  // namespace caching

#endif  // CACHING_H
