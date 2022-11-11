#include "benchmark_params.h"

#include "benchmark_queues.cuh"

#include "queues/broker_queue.cuh"
#include "queues/broker_queue_fast.cuh"
#include "queues/cas_ringbuffer.cuh"
#include "queues/cas_ringbuffer_request.cuh"

#include "utils.cuh"

#include <map>

using DataType = int64_t;
constexpr size_t ringbuffer_size = 128 *  BLOCKSIZE;

using BrokerQueue = queues::BrokerQueue<DataType, ringbuffer_size>;
using BrokerQueueFast = queues::BrokerQueueFast<DataType, ringbuffer_size>;
using CASRingBuffer = queues::CASRingBuffer<DataType, -1, ringbuffer_size>;
using CASRingBufferRequest = queues::CASRingBufferRequest<DataType, -1, ringbuffer_size>;


std::map<std::string, std::function<QueueBenchmark*(void)> > benchmarks = {
    {"BrokerQueue", []() { return new CpuGpuQueueBenchmark<BrokerQueue>("BrokerQueue"); }},
    {"BrokerQueueFast", []() { return new CpuGpuQueueBenchmark<BrokerQueueFast>("BrokerQueueFast"); }},
    {"CASRingBuffer", []() { return new CpuGpuQueueBenchmark<CASRingBuffer>("CASRingBuffer"); }},
    {"CASRingBufferRequest", []() { return new CpuGpuQueueBenchmark<CASRingBufferRequest>("CASRingBufferRequest"); }},
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
        QueueBenchmark* bench= it->second();
        if (bench->CanRunOnGpu()) {
            benchmark::runGpuBenchmark(bench, params.gpu.iterations, params.gpu.blocks, params.gpu.threads);
        }
        if (bench->CanRunOnCpu()) {
            benchmark::runCpuBenchmark(bench, params.cpu.iterations, params.cpu.threads);
        }
    }
        
    

}