#ifndef CAS_RINBUFFER_REQUEST_QUEUE_H
#define CAS_RINBUFFER_REQUEST_QUEUE_H

#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../cuda_utils.cuh"

#include <functional>

namespace queues {


template <typename T, T EMPTY_VALUE, size_t SIZE>
class CASRingBufferRequest {
public:
  typedef T data_type;
  static constexpr bool can_run_on_gpu = true;
  static constexpr bool can_run_on_cpu = false;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");
  static_assert(sizeof(T) == 8, "Size of T must be equal to 8 bytes");

  __device__ CASRingBufferRequest() : head_(0), tail_(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].data = EMPTY_VALUE;
    }
  } 
  __device__ ~CASRingBufferRequest() {}
  

  __device__ bool push(T value, bool insert) {

    constexpr int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];
    __shared__ T requests[warp_count * WARPSIZE];
    __shared__ bool result[warp_count * WARPSIZE];
    int pos = 1;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(pos, pos);
    requests[warp_id * WARPSIZE + pos] = value;
    __syncwarp();

    if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
      for (int i = 0; i <= pos; i++) {
        bool done = false;
        bool appended = false;
        while (!done) {
          if (head_ == tail_ + SIZE) {
            done = true;
          } else {
            uint32_t cur_head = head_;
            T old_value = static_cast<T>(atomicCAS(
              reinterpret_cast<unsigned long long*>(&buffer[cur_head % SIZE].data), 
              static_cast<unsigned long long>(EMPTY_VALUE), 
              static_cast<unsigned long long>(requests[warp_id * WARPSIZE + i])));
            if (old_value == EMPTY_VALUE) {
              appended = true;
              done = true;
            } 
            atomicCAS(const_cast<unsigned*>(static_cast<volatile unsigned*>(&head_)), 
                  static_cast<unsigned>(cur_head), 
                  static_cast<unsigned>(cur_head + 1));
            
          }
        }
        result[warp_id * WARPSIZE + i] = appended;
      } 
    }
    __syncwarp();
    return result[warp_id * WARPSIZE + pos];
  }

  __device__ bool pop(T* res, bool remove) {
    constexpr int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;
    
    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];
    __shared__ T res_values[warp_count * WARPSIZE];
    __shared__ bool result_bools[warp_count * WARPSIZE];
    int pos = 1;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(pos, pos);
    
    

    if (threadIdx.x % WARPSIZE == WARPSIZE - 1) {
      for (int i = 0; i <= pos; i++) { 
        bool done = false;
        result_bools[warp_id * WARPSIZE + i] = false;
        while (!done) {
          if (tail_ == head_) {
            done = true;
          } else {
            uint32_t cur_tail = tail_;
            T result_value = buffer[cur_tail % SIZE].data;
            if (result_value != EMPTY_VALUE) {
              T old_value = static_cast<T>(atomicCAS(
                reinterpret_cast<unsigned long long*>(&buffer[cur_tail % SIZE].data), 
                static_cast<unsigned long long>(result_value), 
                static_cast<unsigned long long>(EMPTY_VALUE)));
              if (old_value == result_value) {
                res_values[warp_id * WARPSIZE + i] = result_value;
                result_bools[warp_id * WARPSIZE + i] = true;
                done = true;
              }
            } 
            // Lets help the other thread
            atomicCAS(const_cast<unsigned*>(static_cast<volatile unsigned*>(&tail_)), 
                static_cast<unsigned>(cur_tail), 
                static_cast<unsigned>(cur_tail + 1));
          }
        }
      }
    }
    __syncwarp();
    *res = res_values[warp_id * WARPSIZE + pos];
    return result_bools[warp_id * WARPSIZE + pos];
  }

private:
  struct Node {
    T data;
  };

  Node buffer[SIZE];
  volatile uint32_t head_;
  volatile uint32_t tail_;
};

}

#endif // !BROKER_QUEUE_H
