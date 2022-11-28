#include <map>

#include "OriginalBrokerQueue.cuh"
#include "benchmark_atomics.cuh"
#include "benchmark_caching.cuh"
#include "benchmark_params.h"
#include "benchmark_queues.cuh"
#include "benchmark_utils.cuh"
#include "benchmark_graphs.cuh"

int main(int argc, char* argv[]) {
  benchmark::BenchParams params = benchmark::parseArguments(argc, argv);

  if (params.atomicadd || params.atomiccas) {
    runAtomicsBenchMark(params);
  }

  if (params.caching) {
    runCachingBenchMark(caching_map, params);
  }

  if (params.queues.size() > 0) {
    runQueueBenchMark(params);
  }

  if (params.graph_name != "") {
    runGraphBenchMark(params);
  }
}