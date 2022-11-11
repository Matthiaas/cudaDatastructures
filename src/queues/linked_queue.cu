#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "../benchmark/utils.cuh"
#define WARPSIZE 32
#define BLOCKSIZE 1024 

template<class T>
struct Node {
    Node<T>* next;
    T data;
    __device__ Node(T&& data) : data(std::forward<T>(data)), next(nullptr) {}
};


template<class T>
struct Queue {
    volatile Node<T>* head;
    volatile Node<T>* tail;
    __device__ __host__ Queue() {
        head = new Node<T>(T());
        tail = head;
    }
};

__device__ Queue<int>* dev_queue;

template<class T>
__device__ void enqueue(volatile Queue<T>* queue, T&& data, bool insert) {
    if (!insert) {
        return;
    }
    volatile Node<T>* node = new Node<T>(std::forward<T>(data));
    int done = 0;
    while(!done) {
        if (atomicCAS(
                (unsigned long long*) &queue->tail->next, 
                (unsigned long long) 0, 
                (unsigned long long) node) == (unsigned long long) 0) { 
            queue->tail = node;           
            done = 1;
        }
    }
}

template<class T>
__device__ void enqueueAsRequest(volatile Queue<T>* queue, T&& data, bool insert) {
    const int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

    __shared__ T requests[warp_count * WARPSIZE];

    int pos;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(insert, pos);
    if (insert) {
        requests[warp_id * WARPSIZE + pos] = std::forward<T>(data);
    }
    __syncwarp();
    if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
        for (int i = 0; i < pos; i++) {
            enqueue(queue, std::forward<T>(requests[warp_id * WARPSIZE + i]), true);
        }
    }
    
}

template<class T>
__device__ void enqueue2(volatile Queue<T>* queue, volatile Node<T>* node) {
    if (node == nullptr) {
        return;
    }
    
    int done = 0;
    while(!done) {
        if (atomicCAS(
                (unsigned long long*) &queue->tail->next, 
                (unsigned long long) 0, 
                (unsigned long long) node) == (unsigned long long) 0) { 
            queue->tail = node;           
            done = 1;
        }
    }
}

template<class T>
__device__ void enqueueAsRequest2(volatile Queue<T>* queue, T&& data, bool insert) {
    // enqueue(queue, std::forward<T>(data), true);
    const int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

    __shared__ volatile Node<T>* requests[warp_count * WARPSIZE];

    int pos;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(insert, pos);
    if (insert) {
        volatile Node<T>* node = new Node<T>(std::forward<T>(data));
        requests[warp_id * WARPSIZE + pos] = node;
    }
    __syncwarp();
    if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
        for (int i = 0; i < pos; i++) {
            enqueue2(queue, requests[warp_id * WARPSIZE + i]);
        }
    }
    
}





template <class T>
__device__ bool dequeue(volatile Queue<T>* queue, T* res) {
    volatile Node<T>* head;
    Node<T>* next;
    T data;
    while(1) {
        head = queue->head;
        if (head->next == nullptr) {
            return false;
        }
        next = head->next;
        data = next->data;
        if (atomicCAS(
                (unsigned long long*) &queue->head, 
                (unsigned long long) head, 
                (unsigned long long) next) == (unsigned long long)head) {
            break;
        }
        
    }
    *res = std::move(data);
    delete head;
    return true;
}

template <class T>
__device__ bool dequeueAsRequest(volatile Queue<T>* queue, T* res) {
    const int warp_count = BLOCKSIZE / WARPSIZE;
    int warp_id = threadIdx.x / WARPSIZE;

    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[warp_count];

    __shared__ T* requests[warp_count * WARPSIZE];
    __shared__ bool results[warp_count * WARPSIZE];

    int pos;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(res != nullptr, pos);

    if (res != nullptr) {
        requests[warp_id * WARPSIZE + pos] = res;
    }
    __syncwarp();

    if(threadIdx.x % WARPSIZE == WARPSIZE - 1) {
        for (int i = 0; i < pos; i++) {
            results[warp_id * WARPSIZE + i] = dequeue(queue, requests[warp_id * WARPSIZE + i]);
        }
    }

    __syncwarp();

    return res == nullptr || results[threadIdx.x];

    

}


__global__ void runQueue(int iterations) {
    for (int i = 0; i < iterations; i++) {
        int res = 0;
        enqueue(dev_queue, 1, true);
        dequeue(dev_queue, &res);
    }
}

__global__ void runQueueAsRequest(int iterations) {
    for (int i = 0; i < iterations; i++) {
        int res = 0;
        enqueueAsRequest(dev_queue, 1, true);
        dequeueAsRequest(dev_queue, &res);
    }
}

__global__ void createQueue() {
    dev_queue = new Queue<int>();
    enqueue(dev_queue, 1, true);
    enqueue(dev_queue, 1, true);
    enqueue(dev_queue, 1, true);
    printf("Queue created\n");
}



// __global__ void testQueue(int iterations) {
//     int tid = threadIdx.x;
//     __shared__ volatile Queue<int>* queue; 
//     if (tid == 1 && blockDim.x == 1 && gridDim.x == 1) {
//         queue = new Queue<int>();
//     }
//     __syncthreads();
    
//     if (tid == 0) {
//         enqueue(queue, 1, true);
//         enqueue(queue, 1, true);
//         enqueue(queue, 1, true);
//     }

//     __syncthreads();


//     for (int i = 0; i < iterations; i++) {
//         int res = 0;
//         enqueue(queue, 1, true);
//         bool deq = dequeue(queue, &res);
//         if (!deq || res != 1) {
//             printf("wrong deq %d\n", res);
//         }
//     }
    

//     __syncthreads();

//     if (tid == 1) {
//         int res;
//         for (int i = 0; i < 3; i++) {
//             bool deq = dequeue(queue, &res);
//             if (!deq || res != 1) {
//                 printf("wrong deq %d\n", res);
//             }
//         }
//     }

// }


int main() {
    int iterations = 10;

    int blocks = 512;
    int threads = BLOCKSIZE;


    auto map = std::map<std::string, std::function<void()>>{
        {"runQueue", [=]() {
            runQueue<<<blocks, threads>>>(iterations);
            cudaDeviceSynchronize();
        }},
        {"runQueueAsRequest", [=]() {
            runQueueAsRequest<<<blocks, threads>>>(iterations);
            cudaDeviceSynchronize();
        }}
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











