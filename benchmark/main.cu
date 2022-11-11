#include "benchmark_params.h"

#include "benchmark_queues.cuh"

#include "queues/broker_queue.cuh"
#include "queues/cas_ringbuffer.cuh"
#include "queues/cas_ringbuffer_request.cuh"

#include "utils.cuh"

#include <map>

using DataType = int64_t;
constexpr size_t ringbuffer_size = 128 *  BLOCKSIZE;

using BrokerQueue = queues::BrokerQueue<DataType, ringbuffer_size>;
using CASRingBuffer = queues::CASRingBuffer<DataType, -1, ringbuffer_size>;
using CASRingBufferRequest = queues::CASRingBufferRequest<DataType, -1, ringbuffer_size>;


std::map<std::string, std::function<QueueBenchmark*(void)> > benchmarks = {
    {"BrokerQueue", []() { return new GPUQueueBenchmark<BrokerQueue>("BrokerQueue"); }},
    {"CASRingBuffer", []() { return new GPUQueueBenchmark<CASRingBuffer>("CASRingBuffer"); }},
    {"CASRingBufferRequest", []() { return new GPUQueueBenchmark<CASRingBufferRequest>("CASRingBufferRequest"); }},
   };


int main(int argc, char *argv[]) {
    benchmark::BenchParams params = benchmark::parseArguments(argc, argv);

    if (params.queues.size() == 0) {
        std::cout << "No queues specified!" << std::endl;
    }
    for (const auto& queuename : params.queues) {
        auto it = benchmarks.find(queuename);
        if (it == benchmarks.end()) {
            std::cerr << "Unknown queue: " << queuename << std::endl;
            return 1;
        }
        benchmark::runGpuBenchmark(it->second(), params.gpu.iterations, params.gpu.blocks, params.gpu.threads);
        benchmark::runCpuBenchmark(it->second(), params.cpu.iterations, params.cpu.threads);
    }
        
    

}