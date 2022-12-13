#include "OriginalBrokerQueue.cuh"
#include "QueueBenchmark.cuh"
#include "benchmark_queues.cuh"
#include "queues/broker_queue.cuh"
#include "queues/broker_queue_fast.cuh"
#include "queues/cas_ringbuffer.cuh"
#include "queues/cas_ringbuffer_request.cuh"
#include "queues/lock_ringbuffer.h"

template <size_t ringbuffer_size>
struct Queues {
  using DataType = int64_t;
  // static constexpr size_t ringbuffer_size = 128 * BLOCKSIZE;

  using BrokerQueue = queues::BrokerQueue<DataType, ringbuffer_size>;
  using BrokerQueueFast = queues::BrokerQueueFast<DataType, ringbuffer_size>;

  static constexpr size_t MAX_THREADS = 1024 * 1024;
  using OriginalBrokerQueue =
      queues::OriginalBrokerQueue<ringbuffer_size, DataType,
                                  MAX_THREADS>;
  using CASRingBuffer = queues::CASRingBuffer<DataType, -1, ringbuffer_size>;
  using CASRingBufferRequest =
      queues::CASRingBufferRequest<DataType, -1, ringbuffer_size>;

  using LockRingBuffer = queues::LockRingBuffer<DataType, ringbuffer_size>;
};


template <size_t ringbuffer_size>
struct Map {
  // This is a hack since CUDA does not support templated globals.
  std::map<std::string, std::function<QueueBenchmark*(void)> > benchmarks = {
    {"BrokerQueue", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::BrokerQueue>("BrokerQueue"); }},
    {"BrokerQueueFast", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::BrokerQueueFast>("BrokerQueueFast"); }},
    {"OriginalBrokerQueue", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::OriginalBrokerQueue>("OriginalBrokerQueue"); }},
    {"CASRingBuffer", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::CASRingBuffer>("CASRingBuffer"); }},
    {"CASRingBufferRequest", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::CASRingBufferRequest>("CASRingBufferRequest"); }},
    {"LockRingBuffer", []() { return new CpuGpuQueueBenchmark<typename Queues<ringbuffer_size>::LockRingBuffer>("LockRingBuffer"); }},
    };
};


template <size_t ringbuffer_size>
void runBenchmark(const benchmark::BenchParams& params) {
  Map<ringbuffer_size> m;
  auto bechmarks_map = m.benchmarks;
  for (const auto& queuename : params.queues) {
    auto it = bechmarks_map.find(queuename);
    if (it == bechmarks_map.end()) {
      std::cerr << "Unknown queue: " << queuename << std::endl;
      return;
    }
    QueueBenchmark* bench = it->second();
    if (bench->CanRunOnGpu() && params.gpu.iterations > 0) {
      benchmark::runGpuBenchmark(bench, params.gpu.iterations,
                                 params.gpu.blocks, params.gpu.threads, ringbuffer_size);
    }
    if (bench->CanRunOnCpu() && params.cpu.iterations > 0) {
      benchmark::runCpuBenchmark(bench, params.cpu.iterations,
                                 params.cpu.threads, ringbuffer_size);
    }
  }
}

void runQueueBenchMark(const benchmark::BenchParams& params) {
    switch (params.ring_buffer_size)
    {
    case 1:
      runBenchmark<1 * 1024>(params);
      break;
    case 2:
      runBenchmark<2 * 1024>(params);
      break;
    case 4:
      runBenchmark<4 * 1024>(params);
      break;
    case 8:
      runBenchmark<8 * 1024>(params);
      break;
    case 16:
      runBenchmark<16 * 1024>(params);
      break;
    case 32:
      runBenchmark<32 * 1024>(params);
      break;
    case 64:
      runBenchmark<64 * 1024>(params);
      break;
    case 128:
      runBenchmark<128 * 1024>(params);
      break;
    case 256:
      runBenchmark<256 * 1024>(params);
      break;
    case 512:
      runBenchmark<512 * 1024>(params);
      break;
    case 1024:
      runBenchmark<1024 * 1024>(params);
      break;
    case 2048:
      runBenchmark<2048 * 1024>(params);
      break;
    case 4096:
      runBenchmark<4096 * 1024>(params);
      break;
    default:
      std::cerr << "Unknown ringbuffer size: " << params.ring_buffer_size << std::endl;
      break;
    }
}

