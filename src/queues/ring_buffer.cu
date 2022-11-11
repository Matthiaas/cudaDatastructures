#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../benchmark/utils.cuh"
#define WARPSIZE 32
#define BLOCKSIZE 1024 


__device__ uint64_t atomicAdd(uint64_t* address, uint64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

__device__ int64_t atomicSub(int64_t* address, int64_t val) {
    return ::atomicAdd((unsigned long long*) address, (unsigned long long) -val);
}


template <typename T, size_t SIZE>
class RingBuffer
{
public:
  __device__ RingBuffer() : head(0), tail(0) {
    for (int i = 0; i < SIZE; i++) {
      buffer[i].state = EMPTY;
    }
  } 
  __device__ ~RingBuffer() {}
  

  __device__ bool push(T value)
  {
    if (count >= SIZE) {
      return false;
    }
      
    if(atomicAdd(&count, 1) >= SIZE) {
      atomicSub(&count, 1);
      return false;
    }

    uint64_t pos = atomicInc(&head, SIZE - 1);
    // printf("push %d at %llu\n", value, pos);
    volatile Node& node = buffer[pos];
    bool loop = true;
    while(loop) {
      if (node.state == EMPTY) {
        node.data = value;
        __threadfence();
        node.state = FULL;
        loop = false;
        // printf("push %d at %llu\n", value, pos);
      }
      // printf("waiting for push %d at %llu\n", value, pos);
    }
    return true;
  }
    

  

  __device__ bool pop(T* res)
  {
    if (count > 0)
    {
      if (atomicSub(&count, 1) <= 0) {
        atomicAdd(&count, 1); 
        return false;
      }
      uint64_t pos = atomicInc(&tail, SIZE - 1);
      // printf("pop pos: %llu\n", pos);
      volatile Node& node = buffer[pos];
      bool loop = true;
      while(loop) {
        if (node.state == FULL) {
          *res = node.data;
          __threadfence();
          node.state = EMPTY;
          loop = false;
        }
        // printf("waiting for pop at %llu\n", pos);
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
    NodeState state;
  };

  volatile Node buffer[SIZE];
  uint32_t head;
  uint32_t tail;
  int64_t count;
};

__device__ RingBuffer<int, 2 * BLOCKSIZE>* buffer;

__global__ void createQueue() {
    buffer = new RingBuffer<int, 2 * BLOCKSIZE>();
    printf("Queue created\n");

    // int64_t *count = new int64_t(0);
    // atomicSub(count, 1);
    // printf("count:  %lld\n", *count);
    // atomicAdd(count, 1);
    // printf("count:  %lld\n", *count);
}
__global__ void runQueue(int iterations) {
    for (int i = 0; i < iterations; ) {
        int res = 0;
        if(buffer->push(i)) {
          buffer->pop(&res);
          i++;
        }
        // printf("poped %d \n", res );
    }
}

int main() {
    int iterations = 10;

    int blocks =16;
    int threads = BLOCKSIZE;


    auto map = std::map<std::string, std::function<void()>>{
        {"runQueue", [=]() {
            runQueue<<<blocks, threads>>>(iterations);
            cudaDeviceSynchronize();
        }},
    };

    auto intialize = []() {
      createQueue<<<1, 1>>>();
      cudaDeviceSynchronize();
    };
    
    auto validate = []() {
        return true;
    };
    timeKernels(intialize, map, validate);
    cudaDeviceSynchronize();
    
}


