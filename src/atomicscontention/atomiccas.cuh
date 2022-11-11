#ifndef ATOMICCAS_H
#define ATOMICCAS_H

#include "cuda_runtime.h"


#define WARPSIZE 32
#define BLOCKSIZE 1024

namespace atomiccas {

__global__ void add_as_accumuluated_requests(int *v);
__global__ void add_as_requests(int *v);
__global__ void add_trival(int *v);

}

#endif // ATOMICCAS_H