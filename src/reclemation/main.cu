#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <cub/cub.cuh>

#include "epoch_base.cuh"

struct A
{
    __device__ A() {
        printf("A::A()\n");
    }
    __device__ ~A() {
        printf("A::~A()\n");
    }
};

constexpr int threads = 2;
constexpr int blocks = 1;
constexpr int global_thread_count = threads * blocks;

__device__ EpochBasedReclamation<A, 2>* epoch = nullptr;

__global__  void setupEpochBased() {
    epoch = new EpochBasedReclamation<A, 2>(global_thread_count);
}

__global__  void testEpochBased() {
    
    epoch->EnterCriticalSection();

   {
        A* a = new A();
        epoch->Retire(a);
   }
   {
        A* a = new A();
        epoch->Retire(a);
   } 
    epoch->LeaveCriticalSection();
    
    

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Global epoch: %lld\n", epoch->GetGlobalEpoch());
    }
    printf("Thread %d: %lld\n", threadIdx.x, epoch->GetLocalEpoch());

    epoch->EnterCriticalSection();
    {
        A* a = new A();
        epoch->Retire(a);
    } 
    if(threadIdx.x == 0) {
        epoch->LeaveCriticalSection();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        printf("Global epoch: %lld\n", epoch->GetGlobalEpoch());
    }
    printf("Thread %d: %lld\n", threadIdx.x, epoch->GetLocalEpoch());
    if(threadIdx.x == 1) {
        epoch->EnterCriticalSection();
    }
    epoch->LeaveCriticalSection();

    if (threadIdx.x == 0) {
        printf("Global epoch: %lld\n", epoch->GetGlobalEpoch());
    }
    printf("Thread %d: %lld\n", threadIdx.x, epoch->GetLocalEpoch());

    epoch->EnterCriticalSection();
    epoch->LeaveCriticalSection();
    
    if (threadIdx.x == 0) {
        printf("Global epoch: %lld\n", epoch->GetGlobalEpoch());
    }
    printf("Thread %d: %lld\n", threadIdx.x, epoch->GetLocalEpoch());

}


int main() {
    setupEpochBased<<<1, 1>>>();
    cudaDeviceSynchronize();
    testEpochBased<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}