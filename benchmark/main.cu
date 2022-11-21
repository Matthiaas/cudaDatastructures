#include "benchmark_params.h"

#include "benchmark_queues.cuh"

#include "queues/broker_queue.cuh"
#include "queues/broker_queue_fast.cuh"
#include "queues/cas_ringbuffer.cuh"
#include "queues/cas_ringbuffer_request.cuh"

#include "atomicscontention/atomicadd.cuh"
#include "atomicscontention/atomiccas.cuh"

#include "OriginalBrokerQueue.cuh"

#include "utils.cuh"

#include <map>

using DataType = int64_t;
constexpr size_t ringbuffer_size = 128 *  BLOCKSIZE;

using BrokerQueue = queues::BrokerQueue<DataType, ringbuffer_size>;
using BrokerQueueFast = queues::BrokerQueueFast<DataType, ringbuffer_size>;
using OriginalBrokerQueue = queues::OriginalBrokerQueue<ringbuffer_size, DataType, BLOCKSIZE * BLOCKSIZE>;
using CASRingBuffer = queues::CASRingBuffer<DataType, -1, ringbuffer_size>;
using CASRingBufferRequest = queues::CASRingBufferRequest<DataType, -1, ringbuffer_size>;


std::map<std::string, std::function<QueueBenchmark*(void)> > benchmarks = {
    {"BrokerQueue", []() { return new CpuGpuQueueBenchmark<BrokerQueue>("BrokerQueue"); }},
    {"BrokerQueueFast", []() { return new CpuGpuQueueBenchmark<BrokerQueueFast>("BrokerQueueFast"); }},
    {"OriginalBrokerQueue", []() { return new CpuGpuQueueBenchmark<OriginalBrokerQueue>("OriginalBrokerQueue"); }},
    {"CASRingBuffer", []() { return new CpuGpuQueueBenchmark<CASRingBuffer>("CASRingBuffer"); }},
    {"CASRingBufferRequest", []() { return new CpuGpuQueueBenchmark<CASRingBufferRequest>("CASRingBufferRequest"); }},
   };

template<typename T>
using AtomicsMap = std::map<std::string, std::function<void(size_t, size_t, size_t, T*)>>;

template<typename T>
AtomicsMap<T> atomicadd_map = {
    {"add_as_accumuluated_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v) { 
        atomicadd::add_as_accumuluated_requests<<<blocks, threads>>>(v,iters); 
    }},
    {"add_as_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
        atomicadd::add_as_requests<<<blocks, threads>>>(v,iters); 
    }},
    {"add_trival", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
        atomicadd::add_trival<<<blocks, threads>>>(v,iters); 
    }},
};

template<typename T>
AtomicsMap<T> atomiccas_map = {
    {"add_as_accumuluated_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v) { 
        atomiccas::add_as_accumuluated_requests<<<blocks, threads>>>(v,iters); 
    }},
    {"add_as_requests", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
        atomiccas::add_as_requests<<<blocks, threads>>>(v,iters); 
    }},
    {"add_trival", [=] (size_t threads, size_t blocks, size_t iters, T* v)  { 
        atomiccas::add_trival<<<blocks, threads>>>(v,iters); 
    }},
};


// This could be so easy if we could use C++20 bind front
template<typename T>
void runAtomicBenchMark(AtomicsMap<T> map, 
                        const benchmark::BenchParams& params) {
    T *v;
    cudaMalloc(&v, sizeof(int));                        
    auto init = [&] { cudaMemset(v, 0, sizeof(int)); };

    auto validate = [&] {
        int h_v;
        cudaMemcpy(&h_v, v, sizeof(int), cudaMemcpyDeviceToHost);
        return (h_v == (params.gpu.blocks * params.gpu.threads * params.gpu.iterations));
    };

    for (auto [name, func] : map) {
        init();
        float time = benchmark::timeKernel(std::bind(func, params.gpu.threads, params.gpu.blocks, params.gpu.iterations, v));
        bool validated = validate();
        std::cout << name << "," << params.gpu.blocks << "," << params.gpu.threads << "," << params.gpu.iterations << ","
             <<  (8 * sizeof(T)) << "," << validated << "," << time << std::endl;
    }

    cudaFree(v);
}

int main(int argc, char *argv[]) {

    benchmark::BenchParams params = benchmark::parseArguments(argc, argv);

    
    if (params.atomicadd) {
        bool found = false;
        if (params.data_type.find("64") != std::string::npos) {
            runAtomicBenchMark(atomicadd_map<uint64_t>, params);
            found = true;
        }
        if (params.data_type.find("32") != std::string::npos) {
            runAtomicBenchMark(atomicadd_map<uint32_t>, params);
            found = true;
        }
        if (!found) {
            std::cout << "Invalid data type for atomicadd: " << params.data_type << std::endl;
            return 1;
        }
    }

    if (params.atomiccas) {
        bool found = false;
        if (params.data_type.find("64") != std::string::npos) {
            runAtomicBenchMark(atomiccas_map<uint64_t>, params);
            found = true;
        }
        if (params.data_type.find("32") != std::string::npos) {
            runAtomicBenchMark(atomiccas_map<uint32_t>, params);
            found = true;
        }
        if (!found) {
            std::cout << "Invalid data type for atomicadd: " << params.data_type << std::endl;
            return 1;
        }
    }

    // if (params.atomiccas) {
    //     runAtomicBenchMark(getAtomicCASMap, params);
    // }
    
    for (const auto& queuename : params.queues) {
        auto it = benchmarks.find(queuename);
        if (it == benchmarks.end()) {
            std::cerr << "Unknown queue: " << queuename << std::endl;
            return 1;
        }
        QueueBenchmark* bench= it->second();
        if (bench->CanRunOnGpu() && params.gpu.iterations > 0) {
            benchmark::runGpuBenchmark(bench, params.gpu.iterations, params.gpu.blocks, params.gpu.threads);
        }
        if (bench->CanRunOnCpu() && params.cpu.iterations > 0) {
            benchmark::runCpuBenchmark(bench, params.cpu.iterations, params.cpu.threads);
        }
    }
        
    

}