#ifndef BROKER_QUEUE_FAST_H
#define BROKER_QUEUE_FAST_H

#include <iostream>

// make -j10 && ./benchmark -queues BrokerQueue -gpu_threads 1024 -gpu_blocks 1024 -gpu_iterations 4096 > out.txt

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
class BrokerQueueFast {
public:
  typedef T data_type;
  static constexpr bool can_run_on_gpu = true;
  static constexpr bool can_run_on_cpu = false;

  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert((SIZE & (SIZE - 1)) == 0, "Size must be a power of 2");

  __device__ BrokerQueueFast() : head(0), tail(0), count(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].ticket = 0;
    }
  } 
  __device__ ~BrokerQueueFast() {
  }

  typedef cub::WarpScan<uint32_t> WarpScan;
  
  __device__ bool push(T value, bool insert)
  {
    constexpr int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    uint32_t participate_in = insert; 
    bool success = insert;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];
    uint32_t participate;  
    WarpScan(temp_storage[warp_id]).ExclusiveSum(participate_in, participate);
    int64_t sum = participate + participate_in;

    __shared__ bool try_again[warp_count];
    if (threadIdx.x % WARPSIZE  == WARPSIZE - 1) {
      if(fetch_and_add(&count, sum) + sum > SIZE) {
        fetch_and_add(&count, -sum);
        try_again[warp_id]  = true;
      } else {
        try_again[warp_id]  = false;
      }
    }
    __syncwarp();
    if (try_again[warp_id] ) {
      if (participate_in) {
        if(fetch_and_add(&count, 1) >= SIZE) {
          fetch_and_add(&count, -1);
          participate_in = 0;
          success = false;
        }
      }
      WarpScan(temp_storage[warp_id]).ExclusiveSum(participate_in, participate);
      sum = participate + participate_in;
    } 
    __shared__ uint64_t shared_tail[warp_count];
    if (threadIdx.x % WARPSIZE  == WARPSIZE - 1) {
      shared_tail[warp_id] = fetch_and_add(&tail, static_cast<uint64_t>(sum));
    }
    __syncwarp();

    if (!success) {
      return false;
    }

    uint64_t my_tail = shared_tail[warp_id] + participate;
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

  __device__ bool pop(T* res, bool remove)
  {
    constexpr int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    uint32_t participate_in = remove; 
    bool success = remove;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];
    uint32_t participate;  
    WarpScan(temp_storage[warp_id]).ExclusiveSum(participate_in, participate);
    int64_t sum = participate + participate_in;
    __shared__ bool try_again[warp_count];
    if (threadIdx.x % WARPSIZE == WARPSIZE - 1) {
      if(fetch_and_add(&count, -sum) - sum < 0) {
        fetch_and_add(&count, sum);
        try_again[warp_id] = true;
      } else {
       
        try_again[warp_id] = false;
      }
    }
    __syncwarp();
    if (try_again[warp_id]) {
       
      if (participate && fetch_and_add(&count, -1) <= 0) {
        fetch_and_add(&count, 1); 
        success = false;
        participate_in = 0;
      }
      WarpScan(temp_storage[warp_id]).ExclusiveSum(participate_in, participate);
      sum = participate + participate_in;
    }

    __shared__ uint64_t shared_head[warp_count];
    if (threadIdx.x % WARPSIZE  == WARPSIZE - 1) {
      
      shared_head[warp_id] = fetch_and_add(&head, static_cast<uint64_t>(sum));
    }
    __syncwarp();

    if (!success) {
      return false;
    }

    uint64_t my_head = shared_head[warp_id] + participate;
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
  using CounterType = typename std::conditional<
    CurrentPlatform == Platform::GPU, 
    int64_t, std::atomic<int64_t>>::type;
    
  using HeadTailType = typename std::conditional<
  CurrentPlatform == Platform::GPU, 
    uint64_t, std::atomic<uint64_t>>::type;


  HeadTailType head;
  HeadTailType tail;
  CounterType count;
};

}

#endif // !BROKER_QUEUE_H
