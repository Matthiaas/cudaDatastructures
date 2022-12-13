#ifndef QUEUE_BENCHMARK_H
#define QUEUE_BENCHMARK_H

#include <thread>

#include "../benchmark/benchmark_utils.cuh"

template <typename T>
__global__ void initQueue(T** buffer) {
  *buffer = new T();
}

template <typename T>
__global__ void runQueue(T** buf, int iterations, int threads_per_block) {
  T* buffer = *buf;
  __shared__ uint32_t done_count;
  if (threadIdx.x == 0) {
    done_count = 0;
  }

  __syncthreads();
  for (int i = 0; done_count != threads_per_block;) {
    // printf("start\n");
    typename T::data_type res = 0;
    bool inserted = buffer->push(i, i < iterations);

    buffer->pop(&res, inserted);
    if (inserted) {
      i++;
      if (i == iterations) {
        atomicAdd(&done_count, 1);
      }
    } 
    __syncthreads();

    // printf("done\n");
  }
}

template <typename T>
__global__ void cleanupQueue(T** buffer) {
  delete *buffer;
}

class QueueBenchmark {
 public:
  QueueBenchmark(const std::string& name) : name_(name) {}
  QueueBenchmark(std::string&& name) : name_(name) {}

  virtual bool CanRunOnGpu() = 0;
  virtual void GpuInit() = 0;
  virtual void GpuRun(size_t num_iter, size_t num_blocks,
                      size_t num_threads) = 0;
  virtual void GpuCleanup() = 0;

  virtual bool CanRunOnCpu() = 0;
  virtual void CpuInit() = 0;
  virtual void CpuRun(size_t num_iter, size_t num_threads) = 0;
  virtual void CpuCleanup() = 0;

  const std::string& name() { return name_; }

 private:
  std::string name_;
};

template <typename T>
struct CpuGpuQueueBenchmark : public QueueBenchmark {
  CpuGpuQueueBenchmark(const std::string& name) : QueueBenchmark(name) {}
  __host__ CpuGpuQueueBenchmark(std::string&& name) : QueueBenchmark(name) {}

  virtual bool CanRunOnGpu() override { return T::can_run_on_gpu; }

  virtual void GpuInit() override {
    if constexpr (T::can_run_on_gpu) {
      cudaMalloc(&d_buffer, sizeof(T*));
      initQueue<<<1, 1>>>(d_buffer);
      cudaDeviceSynchronize();
    }
  }

  virtual void GpuRun(size_t num_iter, size_t num_blocks,
                      size_t num_threads) override {
    if constexpr (T::can_run_on_gpu) {
      runQueue<<<num_blocks, num_threads>>>(d_buffer, num_iter, num_threads);
      cudaDeviceSynchronize();
    }
  }

  virtual void GpuCleanup() override {
    if constexpr (T::can_run_on_gpu) {
      cleanupQueue<<<1, 1>>>(d_buffer);
      cudaDeviceSynchronize();
      cudaFree(d_buffer);
    }
  }

  virtual bool CanRunOnCpu() override { return T::can_run_on_cpu; }

  virtual void CpuInit() override { buffer = new T(); }

  virtual void CpuRun(size_t num_iter, size_t num_threads) override {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
      threads.push_back(std::thread([this, num_iter]() {
        for (int i = 0; i < num_iter;) {
          typename T::data_type res = 0;
          if (buffer->push(i, true)) {
            buffer->pop(&res, true);
            i++;
          }
        }
      }));
    }

    for (auto& t : threads) {
      t.join();
    }
  }

  virtual void CpuCleanup() override { delete buffer; }

 private:
  T** d_buffer;
  T* buffer;
};


#endif  // QUEUE_BENCHMARK_H
