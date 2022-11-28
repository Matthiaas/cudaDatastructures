
#include "default_graph.h"

#include <fstream>
#include <iostream>

DefaultGraph ReadDefaultGraphFromFile(std::string filename) {
  DefaultGraph graph;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << std::endl;
    exit(1);
  }

  file >> graph.num_nodes_;
  graph.num_edges_ = 0;

  graph.graph_.resize(graph.num_nodes_);
  for (size_t i = 0; i < graph.num_nodes_; ++i) {
    uint32_t num_edges;
    file >> num_edges;
    graph.graph_[i].resize(num_edges);
    for (size_t j = 0; j < num_edges; ++j) {
      file >> graph.graph_[i][j];
    }
    graph.num_edges_ += num_edges;
  }

  return graph;
}