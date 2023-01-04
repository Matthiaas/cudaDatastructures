#ifndef CACHING_H
#define CACHING_H

#include <cinttypes>

namespace caching {

__device__ size_t hash_function(std::uint32_t x, uint32_t size) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x % size;
}

__device__ static uint32_t read(volatile uint32_t *addr) { return *addr; }

__global__ void ReadAndWriteSameSlot(uint32_t *data, size_t size,
                                     bool only_one) {
  const size_t idx = threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1) {
      uint32_t pos = hash_function(i, size);
      uint32_t res = read(&data[(pos + idx) % size]);
      // __threadfence();
      if (res == 0)
      if (!only_one || idx == 0) {
        data[pos] = res;
      }
      __threadfence();
    }
  }
}

__global__ void ReadAndWriteDifferentSlot(uint32_t *data, size_t size,
                                          bool only_one) {
  const size_t idx = threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1) {
      uint32_t pos = hash_function(i, size);
      uint32_t res = read(&data[(pos + idx) % size]);
      // 
      if (res == 0)
      if (!only_one || idx == 0) {
        data[(pos + 8) % size] = res;
      }
      __threadfence();
    }
  }
}

__global__ void ReadAndWriteDifferentCacheLine(uint32_t *data, size_t size,
                                               bool only_one) {
  const size_t idx = threadIdx.x;
  if (idx < 8) {
    for (size_t i = 0; i < size; i += 1) {
      uint32_t pos = hash_function(i, size);
      uint32_t res = read(&data[(pos + idx) % size]);
      // __threadfence();
      if (res == 0)
      if (!only_one || idx == 0) {
        data[(pos + 512) % size] = res;
      }
      __threadfence();
    }
  }
}

}  // namespace caching

#endif  // CACHING_H
