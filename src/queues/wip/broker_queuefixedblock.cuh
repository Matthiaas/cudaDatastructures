#ifndef BROKER_QUEUE_H
#define BROKER_QUEUE_H

#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../cuda_utils.cuh"  
#include "../common.h"

#include <functional>
#include <atomic>

namespace queues {

// The BrokerQueue
// https://dl.acm.org/doi/pdf/10.1145/3205289.3205291?casa_token=K6QIArwqwFAAAAAA:VmArpPUZYMBQGTJKLPZnGbYyaBFafszX-kMfXSsLlTTmplen5qG83l3p_yer65b2OB_sHn4wQuT8

template <typename T, size_t SIZE>
class BrokerQueue {
public:
  typedef T data_type;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");

  __device__ __host__ BrokerQueue() : head(0), tail(0), count(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].ticket = 0;
    }
  } 
  __device__ __host__ ~BrokerQueue() {
  }
  
  __device__ bool push(T value, bool insert)
  {
    if (!insert) {
      return false;
    }
    __shared__ bool got_in;
    __shared__ uint64_t shared_tail;
    if (threadIdx.x == 0) {
      if(fetch_and_add(&count, BLOCKSIZE) + BLOCKSIZE > SIZE) {
        fetch_and_add(&count, -BLOCKSIZE);
        got_in = false;
      } else {
        got_in = true;
        shared_tail = fetch_and_add(&tail, BLOCKSIZE);
      }
    }
    __syncthreads();
    if (!got_in) {
      return false;
    }
    uint64_t my_tail = shared_tail + threadIdx.x;
    uint64_t pos = my_tail % SIZE;
    uint64_t ticket = 2 * (my_tail / SIZE);
    volatile Node& node = buffer[pos];
    bool loop = true;
    while(loop) {
      
      if (node.ticket == ticket) {
        node.data = value;
        platformMemFence();
        node.ticket = 2 * (my_tail / SIZE) + 1;
        loop = false;
      }
    }
    return true;
  }

  __device__ bool pop(T* res)
  {
    __shared__ bool got_in;
    __shared__ uint64_t shared_head;
    if (threadIdx.x == 0) {
      if (fetch_and_add(&count, -BLOCKSIZE) - BLOCKSIZE < 0) {
        fetch_and_add(&count, BLOCKSIZE); 
        got_in = false;
      } else {
        got_in = true;
        shared_head = fetch_and_add(&head, BLOCKSIZE);
      }
    }
    __syncthreads();
    if (!got_in) {
      return false;
    }
    uint64_t my_head = shared_head + threadIdx.x;
    uint64_t pos = my_head % SIZE;
    uint64_t ticket = 2 * (my_head / SIZE) + 1;
    volatile Node& node = buffer[pos];
    bool loop = true;
    while(loop) {
      if (node.ticket == ticket) {
        *res = node.data;
        platformMemFence();
        node.ticket = 2 * ((my_head + SIZE) / SIZE);
        loop = false;
      }
    }
    return true;
  }

private:
  enum NodeState {
    EMPTY = 0,
    FULL = 1,
  };
  struct Node {
    T data;
    uint64_t ticket;
  };

  volatile Node buffer[SIZE];
  using CounterType = typename enable_if_else<
    CurrentPlatform == Platform::GPU, 
    int64_t, std::atomic<int64_t>>::type;
    
  using HeadTailType = typename enable_if_else<
  CurrentPlatform == Platform::GPU, 
    uint64_t, std::atomic<uint64_t>>::type;


  HeadTailType head;
  HeadTailType tail;
  CounterType count;
};

}

#endif // !BROKER_QUEUE_H
