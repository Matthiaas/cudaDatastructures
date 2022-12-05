#!/bin/bash

RESULT_FILE="results/real_world_bfs4.txt"
echo "" > $RESULT_FILE

gpu_threads=512
gpu_blocks=512

graphs=( GERMAN_ROAD_NETWORK COMP_SCIENCE_AUTHORS GOOGLE_CONTEST HEP_LITERATURE WWW_NOTRE_DAME US_PATENTS)
for graph_name in "${graphs[@]}" 
do
echo ./build/benchmark -graph_name $graph_name -graph_algos gunrock,bfs,bfs_sharework,bfs_iterations_based -graph_layouts COO,CSR -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks
./build/benchmark -graph_name $graph_name  -graph_algos gunrock,bfs,bfs_sharework,bfs_iterations_based -graph_layouts COO,CSR -gpu_threads $gpu_threads  -gpu_blocks $gpu_blocks >> $RESULT_FILE
done




