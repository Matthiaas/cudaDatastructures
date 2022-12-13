# !/bin/bash

TOTAL_ADDS=$((1024*1024))
RESULT_FILE="results/atomiccas_singleblock.txt"
echo "" > $RESULT_FILE

for (( c=1; c<=1024; c = c * 2 ))
do
    
    gpu_threads=$c
    gpu_blocks=1
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 32
    ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 32 >> $RESULT_FILE
done

for (( c=1; c<=1024; c = c * 2 ))
do
    
    gpu_threads=$c
    gpu_blocks=1
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 64
    ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 64 >> $RESULT_FILE
done


TOTAL_ADDS=$((1024*1024))
RESULT_FILE="results/atomiccas.txt"
echo "" > $RESULT_FILE

for (( gpu_threads=32; gpu_threads<=1024; gpu_threads = gpu_threads * 2 ))
do


    for (( c=1; c<=512; c = c * 2 ))
    do
        gpu_blocks=$c
        gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
        echo ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 32
        ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 32 >> $RESULT_FILE
    done

    for (( c=1; c<=512; c = c * 2 ))
    do
        gpu_blocks=$c
        gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
        echo ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 64
        ./build/benchmark -atomiccas -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations -data_type 64 >> $RESULT_FILE
    done


done

