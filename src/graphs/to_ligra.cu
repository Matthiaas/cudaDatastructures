
#include <iostream>
#include <fstream>
#include <string>

#include "graph_generation/default_graph.h"
#include "ListGraph.cuh"

int main(int argc, char* argv[]) {

  if (argc < 2) {

    std::cout << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
  
    return 0;
  }

  std::string input_file = argv[1];
  std::string output_file = "ligra_" + input_file;
  if (argc > 2) {
    output_file = argv[2];
  } else {
    // Prepend "ligra_" to the input file name but not the path
    size_t last_slash = input_file.find_last_of("/");
    if (last_slash != std::string::npos) {
      output_file = input_file.substr(0, last_slash + 1) + "ligra_" + input_file.substr(last_slash + 1);
    } else {
      output_file = "ligra_" + input_file;
    }
  }

  DefaultGraph graph = ReadDefaultGraphFromFile(input_file);
  ListGraph list_graph = ListGraphFromDefaultGraph(graph);

  std::ofstream output(output_file);

  /*
  The adjacency graph format starts with a sequence of offsets one for each vertex, 
  followed by a sequence of directed edges ordered by their source vertex. 
  The offset for a vertex i refers to the location of the start of a contiguous block 
  of out edges for vertex i in the sequence of edges. The block continues until the offset of the next vertex, 
  or the end if i is the last vertex. All vertices and offsets are 0 based and represented in decimal. 
  The specific format is as follows:

  AdjacencyGraph
  <o0>
  <o1>
  ...
  <o(n-1)>
  <e0>
  <e1>
  ...
  <e(m-1)>
  */

  output << "AdjacencyGraph" << std::endl;
  output << list_graph.num_vertices << std::endl;
  output << list_graph.num_edges << std::endl;

  for (int i = 0; i < list_graph.num_vertices; i++) {
    output << list_graph.offsets[i] << std::endl;
  }

  for (int i = 0; i < list_graph.num_edges; i++) {
    output << list_graph.neighbors[i] << std::endl;
  }

  output.close();
  std::cout << "Written to " << output_file << std::endl;
}