#ifndef BROKER_QUEUE_H
#define BROKER_QUEUE_H

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
class BrokerQueue {
  static const size_t MAX_BLOCKSIZE = 1024;
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

  typedef cub::BlockScan<uint32_t, MAX_BLOCKSIZE> BlockScan;
  
  __device__ bool push(T value, bool insert)
  {
    uint32_t participate_in = insert;
    bool success = insert;

    __shared__ uint64_t shared_tail;
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint32_t participate;  
    BlockScan(temp_storage).ExclusiveSum(participate_in, participate);
    uint32_t sum = participate + participate_in;

    __shared__ bool try_again;
    if (threadIdx.x == blockDim.x - 1) {
      // if(fetch_and_add(&count, sum) + sum > SIZE) {
      //   fetch_and_add(&count, -sum);
      //   try_again = true;
      // } else {
      //   try_again = false;
      // }
      try_again = true;
    }
    __syncthreads();
    if (try_again) {
      if(fetch_and_add(&count, 1) >= SIZE) {
        fetch_and_add(&count, -1);
        participate_in = 0;
        success = false;
      }
      BlockScan(temp_storage).ExclusiveSum(participate_in, participate);
    } 

    if (threadIdx.x == blockDim.x - 1) {
      shared_tail = fetch_and_add(&tail, sum);
    }
    __syncthreads();

    if (!success) {
      return false;
    }

    uint64_t my_tail = shared_tail + participate;
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

  __device__ __host__ bool pop(T* res, bool pop)
  {
    if (!pop) {
      return false;
    }
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

  // __device__ bool pop(T* res, bool remove)
  // {
  //   uint32_t participate_in = remove;
  //   bool success = remove;

  //   __shared__ uint64_t shared_head;
  //   __shared__ typename BlockScan::TempStorage temp_storage;
  //   uint32_t participate;  
  //   BlockScan(temp_storage).ExclusiveSum(participate_in, participate);
  //   uint32_t sum = participate + participate_in;

  //   __shared__ bool try_again;
  //   if (threadIdx.x == BLOCKSIZE - 1) {
  //     // if(fetch_and_add(&count, -sum) - sum < 0) {
  //     //   fetch_and_add(&count, sum);
  //     //   try_again = true;
  //     // } else {
  //     //   try_again = false;
  //     // }
  //     // printf("sum: %d\n", sum);
  //     try_again = true;
  //   }

  //   __syncthreads();

  //   if (try_again) {
  //     if (fetch_and_add(&count, -1) <= 0) {
  //       fetch_and_add(&count, 1); 
  //       participate_in = 0;
  //       success = false;
  //     }
  //     BlockScan(temp_storage).ExclusiveSum(participate_in, participate);
  //   } 

  //   if (threadIdx.x == BLOCKSIZE - 1) {
  //     shared_head = fetch_and_add(&head, sum);
  //   }
  //   __syncthreads();

  //   if (!success) {
  //     return false;
  //   }

  //   uint64_t my_head = shared_head + participate;
  //   uint64_t pos = my_head % SIZE;
  //   uint64_t ticket = 2 * (my_head / SIZE) + 1;
  //   volatile Node& node = buffer[pos];
  //   bool loop = true;
  //   while(loop) {
  //     if (node.ticket == ticket) {
  //       *res = node.data;
  //       platformMemFence();
  //       node.ticket = 2 * ((my_head + SIZE) / SIZE);
  //       loop = false;
  //     }
  //   }
  //   return true;
  // }

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
