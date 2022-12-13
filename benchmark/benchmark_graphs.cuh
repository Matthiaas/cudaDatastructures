#include <iostream>

#include "benchmark_params.h"
#include "graph_generation/default_graph.h"
#include "graphs/Bfs.cuh"
#include "graphs/COOGraph.cuh"
#include "graphs/ListGraph.cuh"

#include "graphs/Gunrock_bfs.cuh"


std::map<std::string, std::map<uint16_t, 
std::function<void(ListGraph,uint32_t*, uint32_t, uint32_t, uint32_t)>>> listGraphAlgorithms = {
    // {"bfs", {
    //   {1, list_graph_aglos::bfs<1, false>},
    //   {2, list_graph_aglos::bfs<2, false>},
    //   {4, list_graph_aglos::bfs<4, false>},
    //   {8, list_graph_aglos::bfs<8, false>},
    //   {16, list_graph_aglos::bfs<16, false>},
    //   {32, list_graph_aglos::bfs<32, false>},
    //   {64, list_graph_aglos::bfs<64, false>},
    //   {128, list_graph_aglos::bfs<128, false>},
    //   {256, list_graph_aglos::bfs<256, false>},
    //   {512, list_graph_aglos::bfs<512, false>},

    // }},
    // {"bfs_sharework", {
    //   {1, list_graph_aglos::bfs<1, true>},
    //   {2, list_graph_aglos::bfs<2, true>},
    //   {4, list_graph_aglos::bfs<4, true>},
    //   {8, list_graph_aglos::bfs<8, true>},
    //   {16, list_graph_aglos::bfs<16, true>},
    //   {32, list_graph_aglos::bfs<32, true>},
    //   {64, list_graph_aglos::bfs<64, true>},
    //   {128, list_graph_aglos::bfs<128, true>},
    //   {256, list_graph_aglos::bfs<256, true>},
    //   {512, list_graph_aglos::bfs<512, true>},
    // }},

    {"bfs", {
      {1, list_graph_aglos::bfs<1, false, false>},
      {2, list_graph_aglos::bfs<2, false, false>},
      {4, list_graph_aglos::bfs<4, false, false>},
      {8, list_graph_aglos::bfs<8, false, false>},
      {16, list_graph_aglos::bfs<16, false, false>},
      {32, list_graph_aglos::bfs<32, false, false>},
      {64, list_graph_aglos::bfs<64, false, false>},
      {128, list_graph_aglos::bfs<128, false, false>},
      {256, list_graph_aglos::bfs<256, false, false>},
      {512, list_graph_aglos::bfs<512, false, false>},

    }},
    {"bfs_sharework", {
      {1, list_graph_aglos::bfs<1, true, false>},
      {2, list_graph_aglos::bfs<2, true, false>},
      {4, list_graph_aglos::bfs<4, true, false>},
      {8, list_graph_aglos::bfs<8, true, false>},
      {16, list_graph_aglos::bfs<16, true, false>},
      {32, list_graph_aglos::bfs<32, true, false>},
      {64, list_graph_aglos::bfs<64, true, false>},
      {128, list_graph_aglos::bfs<128, true, false>},
      {256, list_graph_aglos::bfs<256, true, false>},
      {512, list_graph_aglos::bfs<512, true, false>},
    }},
};


std::map<std::string, std::function<void(COOGraph,uint32_t*, uint32_t, uint32_t, uint32_t)>> cooGraphAlgorithms = {
    {"bfs", coo_graph_aglos::bfs},  
    {"bfs_iterations_based", coo_graph_aglos::bfs_iterations_based},
};

std::map<std::string, std::function<void(ListGraph, int*, int)>> gunrock_bfss = {
    {"gunrock-thread_mapped", gunrock_bfs<gunrock::operators::load_balance_t::thread_mapped> },
    // Not supported
    // {"gunrock warp_mapped", gunrock_bfs<gunrock::operators::load_balance_t::warp_mapped> },
    {"gunrock-block_mapped", gunrock_bfs<gunrock::operators::load_balance_t::block_mapped> },
    // Not supported
    // {"gunrock bucketing", gunrock_bfs<gunrock::operators::load_balance_t::bucketing> },
    {"gunrock-merge_path", gunrock_bfs<gunrock::operators::load_balance_t::merge_path> },
    // Segfaults
    // {"gunrock merge_path_v2", gunrock_bfs<gunrock::operators::load_balance_t::merge_path_v2> },
    // Not supported
    // {"gunrock work_stealing", gunrock_bfs<gunrock::operators::load_balance_t::work_stealing> },
};

void runGraphBenchMark(const benchmark::BenchParams &params) {
  std::string filename = "./generated_graphs/"  + params.graph_name + ".txt";

  DefaultGraph graph = ReadDefaultGraphFromFile(filename );

  uint32_t* distances1;
  cudaMalloc(&distances1, graph.NumNodes() * sizeof(uint32_t));
  CUERR

  auto time_call = [](auto fun) {
    auto time_start = std::chrono::high_resolution_clock::now();
    fun();
    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                       time_end - time_start)
                       .count() /
                   1000.0;
    return time_ms;
  };

  const uint32_t start_node = 0;

  for (const auto& graph_type : params.graph_layouts) {
    for (const auto& algo : params.graph_algos) {
      if (graph_type == "COO") {
        auto it = cooGraphAlgorithms.find(algo);
        if (it == cooGraphAlgorithms.end()) {
          std::cerr << "Algorithm " << algo << " not found for COO"
                    << std::endl;
          continue;
        }
        auto fun = it->second;
        COOGraph coo_graph = COOFromDefaultGraph(graph);
        COOGraph coo_graph_d = coo_graph.CopyToDevice();
        double ms =
            time_call(std::bind(fun, coo_graph_d, distances1, start_node,
                                params.gpu.blocks, params.gpu.threads));
        std::cout << params.graph_name << " COO " << algo << " 1 " << ms
                  << std::endl;
        coo_graph.Free();
        coo_graph_d.Free();
      } else if (graph_type == "CSR") {
        if (algo == "gunrock") {
          ListGraph list_graph = ListGraphFromDefaultGraph(graph);
          ListGraph list_graph_d = list_graph.CopyToDevice();
          
          
          for (const auto& gunrock_algo : gunrock_bfss) {
            double ms = time_call(std::bind(gunrock_algo.second, list_graph_d,
                                          reinterpret_cast<int*>(distances1),
                                          start_node));
            std::cout << params.graph_name << " CSR " << gunrock_algo.first << " "
                        << 0 << " " << ms << std::endl;
          }
          list_graph.Free();
          list_graph_d.Free();
        } else {
          auto it = listGraphAlgorithms.find(algo);
          if (it == listGraphAlgorithms.end()) {
            std::cerr << "Algorithm " << algo << " not found for CSR"
                      << std::endl;
            continue;
          }
          for (const auto group_size : it->second) {
            auto fun = group_size.second;
            ListGraph list_graph = ListGraphFromDefaultGraph(graph);
            ListGraph list_graph_d = list_graph.CopyToDevice();
            double ms =
                time_call(std::bind(fun, list_graph_d, distances1, start_node,
                                    params.gpu.blocks, params.gpu.threads));
            std::cout << params.graph_name << " CSR " << algo << " "
                      << group_size.first << " " << ms << std::endl;
            list_graph.Free();
            list_graph_d.Free();
          }
        }
      }
    }
  }

  cudaFree(distances1); CUERR
}

// int main() {
//   std::string filename = "../generated_graphs/US_PATENTS.txt";

//   DefaultGraph graph = ReadDefaultGraphFromFile(filename);

//   uint32_t *distances1;
//   uint32_t *distances2;
//   cudaMallocManaged(&distances1, graph.NumNodes() * sizeof(uint32_t));
//   CUERR
//   cudaMallocManaged(&distances2, graph.NumNodes() * sizeof(uint32_t));
//   CUERR

//   {
//     COOGraph coo_graph = COOFromDefaultGraph(graph);
//     CUERR
//     COOGraph coo_graph_d = coo_graph.CopyToDevice();
//     CUERR
//     bfs(coo_graph_d, distances1, 0);
//     coo_graph.Free();
//     coo_graph_d.Free();
//   }

//   {
//     ListGraph list_graph = ListGraphFromDefaultGraph(graph);
//     ListGraph list_graph_d = list_graph.CopyToDevice();
//     bfs_sharework<32>(list_graph_d, distances2, 0);
//     list_graph.Free();
//     list_graph_d.Free();
//   }

//   for (uint32_t i = 0; i < graph.NumNodes(); i++) {
//     if (distances1[i] != distances2[i]) {
//       std::cout << "Error at node " << i << std::endl;
//       std::cout << "COO: " << distances1[i] << std::endl;
//       std::cout << "List: " << distances2[i] << std::endl;
//       break;
//     }
//   }

//   cudaFree(distances1);
//   CUERR
//   cudaFree(distances2);
//   CUERR

//   std::cout << "Done" << std::endl;

//   return 0;
// }