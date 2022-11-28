#ifndef DEFAULT_GRAPH_H_
#define DEFAULT_GRAPH_H_

#include <cinttypes>
#include <string>
#include <vector>

class DefaultGraph {
 public:
  DefaultGraph() : num_nodes_(0), num_edges_(0) {}
  size_t NumNodes() const { return num_nodes_; }
  size_t NumEdges() const { return num_edges_; }
  const std::vector<std::vector<uint32_t>>& GetEdgeLists() const {
    return graph_;
  }

  void AddEdge(uint32_t from, uint32_t to) {
    if (from >= num_nodes_) {
      num_nodes_ = from + 1;
      graph_.resize(num_nodes_);
    }
    if (to >= num_nodes_) {
      num_nodes_ = to + 1;
      graph_.resize(num_nodes_);
    }
    graph_[from].push_back(to);
    ++num_edges_;
  }

 private:
  std::vector<std::vector<uint32_t>> graph_;
  size_t num_nodes_;
  size_t num_edges_;  

  friend DefaultGraph ReadDefaultGraphFromFile(std::string filename);
};


DefaultGraph ReadDefaultGraphFromFile(std::string filename);

#endif  // DEFAULT_GRAPH_H_
