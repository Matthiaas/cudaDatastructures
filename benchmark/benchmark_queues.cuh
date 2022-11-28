#ifndef BECHMARK_QUEUES_H
#define BECHMARK_QUEUES_H

#include "QueueBenchmark.cuh"
#include "queues/broker_queue.cuh"
#include "queues/broker_queue_fast.cuh"
#include "queues/cas_ringbuffer.cuh"
#include "queues/cas_ringbuffer_request.cuh"

using DataType = int64_t;
constexpr size_t ringbuffer_size = 128 * BLOCKSIZE;

using BrokerQueue = queues::BrokerQueue<DataType, ringbuffer_size>;
using BrokerQueueFast = queues::BrokerQueueFast<DataType, ringbuffer_size>;
using OriginalBrokerQueue =
    queues::OriginalBrokerQueue<ringbuffer_size, DataType,
                                BLOCKSIZE * BLOCKSIZE>;
using CASRingBuffer = queues::CASRingBuffer<DataType, -1, ringbuffer_size>;
using CASRingBufferRequest =
    queues::CASRingBufferRequest<DataType, -1, ringbuffer_size>;

std::map<std::string, std::function<QueueBenchmark*(void)> > benchmarks = {
  {"BrokerQueue", []() { return new CpuGpuQueueBenchmark<BrokerQueue>("BrokerQueue"); }},
  {"BrokerQueueFast", []() { return new CpuGpuQueueBenchmark<BrokerQueueFast>("BrokerQueueFast"); }},
  {"OriginalBrokerQueue", []() { return new CpuGpuQueueBenchmark<OriginalBrokerQueue>("OriginalBrokerQueue"); }},
  {"CASRingBuffer", []() { return new CpuGpuQueueBenchmark<CASRingBuffer>("CASRingBuffer"); }},
  {"CASRingBufferRequest", []() { return new CpuGpuQueueBenchmark<CASRingBufferRequest>("CASRingBufferRequest"); }},
  };



void runQueueBenchMark(const benchmark::BenchParams& params) {
    for (const auto& queuename : params.queues) {
    auto it = benchmarks.find(queuename);
    if (it == benchmarks.end()) {
      std::cerr << "Unknown queue: " << queuename << std::endl;
      return;
    }
    QueueBenchmark* bench = it->second();
    if (bench->CanRunOnGpu() && params.gpu.iterations > 0) {
      benchmark::runGpuBenchmark(bench, params.gpu.iterations,
                                 params.gpu.blocks, params.gpu.threads);
    }
    if (bench->CanRunOnCpu() && params.cpu.iterations > 0) {
      benchmark::runCpuBenchmark(bench, params.cpu.iterations,
                                 params.cpu.threads);
    }
  }
}

#endif  // BECHMARK_QUEUES_H
