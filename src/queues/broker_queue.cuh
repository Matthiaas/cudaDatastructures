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
  static constexpr bool can_run_on_gpu = true;
  static constexpr bool can_run_on_cpu = true;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");

  __device__ __host__ BrokerQueue() : head(0), tail(0), count(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].ticket = 0;
    }
  } 
  __device__ __host__ ~BrokerQueue() {
  }
  

  __device__ __host__ bool push(T value, bool insert)
  {
    if (count >= SIZE || !insert) {
      return false;
    }
    if(fetch_and_add(&count, 1) >= SIZE) {
      fetch_and_add(&count, -1);
      return false;
    }
    uint64_t my_tail = fetch_and_add(&tail, 1);
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

  __device__ __host__ bool pop(T* res, bool insert)
  {
    if (count > 0)
    {
      if (fetch_and_add(&count, -1) <= 0) {
        fetch_and_add(&count, 1); 
        return false;
      }
      uint64_t my_head = fetch_and_add(&head, 1);
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
    return false;
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
  // TODO: What if this overflows?
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
