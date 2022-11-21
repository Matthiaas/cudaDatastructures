#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// struct A {
//     __device__ A() {
//     }
// };

// template <typename T>
// struct B {
//     B() {
//         A test;
//     }
// };


// // Kernel function to add the elements of two arrays
// __global__
// void add(int n, float *x, float *y)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = index; i < n; i += stride)
//     y[i] = x[i] + y[i];
// }



struct A {
    A() {
      cudaMalloc(&a, 100 * sizeof(int));
    }

    __device__ int getat(int i) {
      return a[i];
    }

    

  int* a; 
};

__global__ void add(A* a)
{
  printf("hi");
  printf("%d\n", a->getat(threadIdx.x));
}

int main(void)
{
  // int N = 1<<20;
  // float *x, *y;

  A a;
  add<<<1, 10>>>(&a);

  cudaDeviceSynchronize();

  


  // // Allocate Unified Memory – accessible from CPU or GPU
  // cudaMallocManaged(&x, N*sizeof(float));
  // cudaMallocManaged(&y, N*sizeof(float));

  // // initialize x and y arrays on the host
  // for (int i = 0; i < N; i++) {
  //   x[i] = 1.0f;
  //   y[i] = 2.0f;
  // }

  // int blockSize = 256;
  //   int numBlocks = (N + blockSize - 1) / blockSize;
  //   add<<<numBlocks, blockSize>>>(N, x, y);

  // // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();

  // // Check for errors (all values should be 3.0f)
  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = fmax(maxError, fabs(y[i]-3.0f));
  // std::cout << "Max error: " << maxError << std::endl;

  // // Free memory
  // cudaFree(x);
  // cudaFree(y);
  
  return 0;
}