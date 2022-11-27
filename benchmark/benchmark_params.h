#ifndef BENCHMARK_PARAMS_H
#define BENCHMARK_PARAMS_H

#include <sstream>
#include <string>
#include <vector>

namespace benchmark {

struct CPUSettings {
  size_t threads;
  size_t iterations;
};

struct GPUSettings {
  size_t threads;
  size_t blocks;
  size_t iterations;
};

struct BenchParams {
  CPUSettings cpu;
  GPUSettings gpu;
  std::vector<std::string> queues;
  bool atomicadd;
  bool atomiccas;
  bool caching;
  std::string data_type;
};

std::vector<std::string> split(const std::string &s, char delim);

BenchParams parseArguments(int argc, char *argv[]);

}  // namespace benchmark

#endif