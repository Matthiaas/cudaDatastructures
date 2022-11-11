#ifndef CAS_RINBUFFER_QUEUE_H
#define CAS_RINBUFFER_QUEUE_H

#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../cuda_utils.cuh"

#include <functional>

namespace queues {

template <typename T, T EMPTY_VALUE, size_t SIZE>
class CASRingBuffer {
public:
  typedef T data_type;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");
  static_assert(sizeof(T) == 8, "Size of T must be equal to 8 bytes");

  __device__ __host__ CASRingBuffer() : head_(0), tail_(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].data = EMPTY_VALUE;
    }
  } 
  __device__ __host__ ~CASRingBuffer() {}
  

  __device__ __host__ bool push(T value, bool insert) {
    bool done = false;
    bool appended = false;
    while (!done) {
      if (head_ == tail_ + SIZE) {
        done = true;
      } else {
        uint32_t cur_head = head_;
        T old_value = static_cast<T>(platformCAS(
          reinterpret_cast<unsigned long long*>(&buffer[cur_head % SIZE].data), 
          static_cast<unsigned long long>(EMPTY_VALUE), 
          static_cast<unsigned long long>(value)));
        if (old_value == EMPTY_VALUE) {
          appended = true;
          done = true;
        } 
        platformCAS(const_cast<unsigned*>(static_cast<volatile unsigned*>(&head_)), 
              static_cast<unsigned>(cur_head), 
              static_cast<unsigned>(cur_head + 1));
        
      }
    }
    return appended;
  }

  __device__ __host__ bool pop(T* res, bool remove) {
    bool done = false;
    bool popped = false;
    while (!done) {
      if (tail_ == head_) {
        done = true;
      } else {
        uint32_t cur_tail = tail_;
        T result = buffer[cur_tail % SIZE].data;
        if (result != EMPTY_VALUE) {
          T old_value = static_cast<T>(platformCAS(
            reinterpret_cast<unsigned long long*>(&buffer[cur_tail % SIZE].data), 
            static_cast<unsigned long long>(result), 
            static_cast<unsigned long long>(EMPTY_VALUE)));
          if (old_value == result) {
            *res = result;
            popped = true;
            done = true;
          }
        } 
        // Lets help the other thread
        platformCAS(const_cast<unsigned*>(static_cast<volatile unsigned*>(&tail_)), 
            static_cast<unsigned>(cur_tail), 
            static_cast<unsigned>(cur_tail + 1));
      }
    }
    return popped;
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
