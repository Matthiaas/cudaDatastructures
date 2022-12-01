cmake_minimum_required(VERSION 3.24)
project(cudaDataStructures LANGUAGES CXX CUDA)



if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)


# Enable Warnings
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic")


# Release Flags
set(CUDA_NVCC_FLAGS_RELEASE "--extended-lambda")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Debug Flags
# 
set(CUDA_NVCC_FLAGS_DEBUG "--lineinfo -g -G --extended-lambda")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -G")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")

include_directories(third_party)
include_directories(src)
include_directories(/usr/local/cuda-11.8/include/)


set(BENCH_SOURCES
    benchmark/benchmark_utils.cuh
    benchmark/benchmark_atomics.cuh
    benchmark/benchmark_graphs.cuh
    benchmark/benchmark_queues.cuh
    benchmark/QueueBenchmark.cuh
    benchmark/benchmark_params.h
    benchmark/benchmark_params.cpp)

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
        third_party/OriginalBrokerQueue.cuh
        )

set(THIRD_PARTY_ARG_PARSER_SOURCES
        third_party/arg_parser.cpp
        third_party/arg_parser.h
        )

set(THIRD_PARTY_GRAPH_SOURCES
        third_party/graph_generation/graph_generation.h
        third_party/graph_generation/graph_generation.cpp
        third_party/graph_generation/graph_generation_util.h
        third_party/graph_generation/graph_generation_util.cpp
)


set(THIRD_PARTY_DEFAULT_GRAPH_SOURCES
        third_party/graph_generation/default_graph.h
        third_party/graph_generation/default_graph.cpp
        )

file(GLOB THIRD_PARTY_DYCUCKOO_SOURCES
        third_party/dycuckoo/dynamicHash/core/*.cuh
        third_party/dycuckoo/dynamicHash/data/*.cuh
        third_party/dycuckoo/dynamicHash/include/*.cuh
        third_party/dycuckoo/dynamicHash/thirdParty/*.h
        third_party/dycuckoo/dynamicHash/thirdParty/cnmem.cpp
        )   

file(GLOB THIRD_PARTY_WARPCORE_SOURCES
        third_party/warpcore/*.cuh
        third_party/warpcore/*.cu
        )   

add_executable(benchmark
    benchmark/main.cu
    ${BENCH_SOURCES}
    ${QUEUE_SOURCES}
    ${ATOMIC_SOURCES}
    ${THIRD_PARTY_SOURCES}
    ${THIRD_PARTY_ARG_PARSER_SOURCES}
    ${THIRD_PARTY_DEFAULT_GRAPH_SOURCES}
    )

add_executable(graph_generator
    ${THIRD_PARTY_GRAPH_SOURCES}
    ${THIRD_PARTY_ARG_PARSER_SOURCES}
    )

add_executable(bench_hash
    benchmark/benchmark_hashtable.cu
    ${THIRD_PARTY_DYCUCKOO_SOURCES}
    ${THIRD_PARTY_WARPCORE_SOURCES}
    third_party/dycuckoo/dynamicHash/thirdParty/cnmem.cpp
)
set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES native)
# set_property(TARGET test PROPERTY CUDA_ARCHITECTURES native)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
target_compile_options(bench_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
--extended-lambda -O3
                       >)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_options(bench_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        ${CUDA_NVCC_FLAGS_DEBUG}
                       >)
endif()