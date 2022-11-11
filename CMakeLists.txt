cmake_minimum_required(VERSION 3.24)
project(cudaDataStructures LANGUAGES CXX CUDA)



if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --use_fast_math -rdc=true -lcudadevrt")

# Enable Warnings
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Werror")

# Release Flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -O3")

# Debug Flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -G")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")

include_directories(third_party)
include_directories(src)


set(BENCH_SOURCES
    benchmark/benchmark_params.h
    benchmark/benchmark_params.cpp
    benchmark/main.cu)

set(QUEUE_SOURCES
        src/queues/broker_queue.cuh
        src/queues/cas_ringbuffer.cuh
        src/queues/cas_ringbuffer_request.cuh
        )

set(ATOMIC_SOURCES
        src/atomicscontention/atomicadd.cu
        src/atomicscontention/atomicadd.cuh
        src/atomicscontention/atomiccas.cu
        src/atomicscontention/atomiccas.cuh
        )

set(THIRD_PARTY_SOURCES
        third_party/arg_parser.cpp
        third_party/arg_parser.h
        )

add_executable(benchmark
    ${BENCH_SOURCES}
    ${QUEUE_SOURCES}
    ${ATOMIC_SOURCES}
    ${THIRD_PARTY_SOURCES}
    )
    

set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES native)