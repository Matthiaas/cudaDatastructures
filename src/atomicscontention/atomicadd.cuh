#ifndef ATOMICADD_H
#define ATOMICADD_H

#include "cuda_runtime.h"

namespace atomicadd {

__global__
void add_as_accumuluated_requests(int *v);

__global__
void add_as_requests(int *v);

__global__
void add_trival(int *v);

}

#endif