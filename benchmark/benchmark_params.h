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
  size_t ring_buffer_size;

  bool atomicadd;
  bool atomiccas;
  std::string data_type;

  bool caching; 

  std::string graph_name;
  std::vector<std::string> graph_algos;
  std::vector<std::string> graph_layouts;
};

std::vector<std::string> split(const std::string &s, char delim);

BenchParams parseArguments(int argc, char *argv[]);

}  // namespace benchmark

#endif