#include "benchmark_params.h"

#include "arg_parser.h"

namespace benchmark {

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::istringstream iss(s);
  std::string item;
  while (std::getline(iss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

BenchParams parseArguments(int argc, char *argv[]) {
  BenchParams params;
  arg_parser parser(argc, argv);

  params.cpu.threads = parser.getCmdOptionAsInt("-cpu_threads").value_or(1);
  params.cpu.iterations =
      parser.getCmdOptionAsInt("-cpu_iterations").value_or(0);
  params.gpu.threads = parser.getCmdOptionAsInt("-gpu_threads").value_or(1);
  params.gpu.blocks = parser.getCmdOptionAsInt("-gpu_blocks").value_or(1);
  params.gpu.iterations =
      parser.getCmdOptionAsInt("-gpu_iterations").value_or(0);
  params.atomicadd = parser.cmdOptionExists("-atomicadd");
  params.atomiccas = parser.cmdOptionExists("-atomiccas");
  params.caching = parser.cmdOptionExists("-caching");
  params.data_type = parser.getCmdOption("-data_type").value_or("none");
  const auto queues = parser.getCmdOption("-queues");

  if (queues.has_value()) {
    params.queues = split(std::string(queues.value()), ',');
  } else {
    params.queues = {};
  }
  return params;
}

}  // namespace benchmark
