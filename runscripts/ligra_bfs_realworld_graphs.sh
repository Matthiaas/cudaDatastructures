#!/bin/bash

cd externals
# git clone  https://github.com/jshun/ligra.git
cd ligra
cd apps

make BFS
g++ -std=c++14 -O3 -DBYTERLE -o BFS BFS.C -fopenmp -DOPENMP

RESULT_FILE="../../../results/ligra_real_world_bfs.txt"
echo "" > $RESULT_FILE

graphs=(GERMAN_ROAD_NETWORK COMP_SCIENCE_AUTHORS GOOGLE_CONTEST HEP_LITERATURE WWW_NOTRE_DAME US_PATENTS)


for (( t=1; t <= 32; t = t * 2 ))
do

for graph_name in "${graphs[@]}" 
do
  echo OMP_NUM_THREADS=$t ./BFS -rounds 32 ../../../generated_graphs/ligra_$graph_name.txt
  RUNTIME=$(OMP_NUM_THREADS=$t ./BFS -rounds 32 ../../../generated_graphs/ligra_$graph_name.txt | grep -o -E '[0-9]+.*')  
  for run_time in $RUNTIME
  do
    echo $graph_name $t $run_time >> $RESULT_FILE
  done
done

done








