#!/bin/bash

RESULT_FILE="results/hash_table_cpu_final.txt"
echo "" > $RESULT_FILE

for (( c=0; c < 32; c = c + 1 ))
do
    
echo ./build/bench_hash 
./build/bench_hash >> $RESULT_FILE

done