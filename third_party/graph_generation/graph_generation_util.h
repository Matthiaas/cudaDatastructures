#ifndef TEAM02_GRAPH_GENERATION_UTIL_H
#define TEAM02_GRAPH_GENERATION_UTIL_H

#include <string>
#include <vector>
#include <ctime>

/*
 * A method that takes as argument a file name and unzips the file if needed.
 * The method returns the file name of the unzipped file.
 */
std::string unzip_file(std::string file_name);

/*
 * A method that takes as argument an input file stream and an adjacency list.
 * Iterates reads in the file line by line, parsing each line into a src and dst
 * vertex and adding it the adjacency list. The method returns the number of
 * edges in the graph.
 */
uint64_t parse_edge_list(std::ifstream &infile,
                         std::vector<std::vector<uint64_t>> &adjacency_list,
                         bool initialized = true);

/*
 * A method that takes as argument a graph definition, an output file stream,
 * and an adjacency list. The method outputs the adjacency list to the output
 * file stream.
 */
void output_adjacency_list(std::ofstream &outfile,
                           std::vector<std::vector<uint64_t>> &adjacency_list,
                           bool shuffle, uint64_t seed = std::time(nullptr));

#endif