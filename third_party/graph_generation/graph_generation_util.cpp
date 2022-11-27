#include "graph_generation_util.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * A method that takes as argument a file name.
 * The method returns true if the file exists, false otherwise.
 */
bool file_exists(const std::string &name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

/*
 * A method that takes as argument a file name and unzips the file if needed.
 * The method returns the file name of the unzipped file.
 */
std::string unzip_file(std::string file_name) {
  if (file_name.find(".gz") != std::string::npos) {
    std::string unzipped_file_name = file_name.substr(0, file_name.find(".gz"));
    if (!file_exists(unzipped_file_name)) {
      std::cout << file_name << " is unzipping..." << std::endl;
      std::string command =
          "gzip -d -c " + file_name + " > " + unzipped_file_name;
      int exit_code = system(command.c_str());
      if (exit_code != 0) {
        throw std::runtime_error("gzip failed");
      }
      std::cout << file_name << " unzipped." << std::endl;
    }
    return unzipped_file_name;
  } else {
    return file_name;
  }
}

/*
 * A method that takes as argument an input file stream and an adjacency list.
 * Iterates reads in the file line by line, parsing each line into a src and dst
 * vertex and adding it the adjacency list. The method returns the number of
 * edges in the graph.
 */
uint64_t parse_edge_list(std::ifstream &infile,
                         std::vector<std::vector<uint64_t>> &adjacency_list,
                         bool initialized) {
  std::unordered_map<uint64_t, uint64_t> nodeid_to_index;
  uint64_t node_id_from, node_id_to;
  uint64_t edges_added = 0, nodes_count = 0;
  std::string line;

  // read in the patent id and patent index line by line
  while (std::getline(infile, line)) {
    // get the two patent ids from line
    std::stringstream ss(line);
    ss >> node_id_from >> node_id_to;
    if (nodeid_to_index.find(node_id_from) == nodeid_to_index.end()) {
      nodeid_to_index[node_id_from] = nodes_count;
      node_id_from = nodes_count;
      nodes_count++;
      if (!initialized) {
        adjacency_list.push_back(std::vector<uint64_t>());
      }
    } else {
      node_id_from = nodeid_to_index[node_id_from];
    }
    if (nodeid_to_index.find(node_id_to) == nodeid_to_index.end()) {
      nodeid_to_index[node_id_to] = nodes_count;
      node_id_to = nodes_count;
      nodes_count++;
      if (!initialized) {
        adjacency_list.push_back(std::vector<uint64_t>());
      }
    } else {
      node_id_to = nodeid_to_index[node_id_to];
    }

    if (node_id_from == node_id_to ||
        std::find(adjacency_list[node_id_from].begin(),
                  adjacency_list[node_id_from].end(),
                  node_id_to) != adjacency_list[node_id_from].end()) {
      continue;
    }
    adjacency_list[node_id_from].push_back(node_id_to);
    adjacency_list[node_id_to].push_back(node_id_from);
    edges_added++;
  }
  return edges_added;
}

/*
 * A method that takes as argument a graph definition, an output file stream,
 * and an adjacency list. The method outputs the adjacency list to the output
 * file stream.
 */
void output_adjacency_list(std::ofstream &outfile,
                           std::vector<std::vector<uint64_t>> &adjacency_list,
                           bool shuffle, uint64_t seed) {
  std::default_random_engine gen(seed);
  outfile << adjacency_list.size() << std::endl;
  for (uint64_t i = 0; i < adjacency_list.size(); i++) {
    outfile << adjacency_list[i].size() << " ";
    if (shuffle) {
      std::shuffle(adjacency_list[i].begin(), adjacency_list[i].end(), gen);
    }
    for (uint64_t j = 0; j < adjacency_list[i].size(); j++) {
      outfile << adjacency_list[i][j] << " ";
    }
    outfile << std::endl;
  }
}