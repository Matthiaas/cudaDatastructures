#!/bin/bash

RESULT_FILE="results/queues.txt"
echo "" > $RESULT_FILE
TOTAL_ADDS=$((1024*1024))
for (( c=1; c<=1024; c = c * 2 ))
do
    
    gpu_threads=$c
    gpu_blocks=1
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -queues OriginalBrokerQueue,BrokerQueue -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations 
    ./build/benchmark -queues OriginalBrokerQueue,BrokerQueue -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_FILE
done

TOTAL_ADDS=$((1024*1024*1024))

for (( c=4096; c>=1; c = c / 2 ))
do
    
    gpu_threads=1024
    gpu_blocks=$c
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./benchmark -queues OriginalBrokerQueue,BrokerQueue,BrokerQueueFast -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations
    ./build/benchmark -queues OriginalBrokerQueue,BrokerQueue,BrokerQueueFast -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_FILE
done


TOTAL_ADDS=$((1024*1024*128))
RESULT_FILE="results/cpu_queues.txt"
echo "" > $RESULT_FILE

for (( c=32; c>=1; c = c / 2 ))
do
    
    cpu_threads=$c
    cpu_blocks=1
    cpu_iterations=$(( TOTAL_ADDS / (cpu_threads * cpu_blocks) ))
    echo ./benchmark -queues BrokerQueue -cpu_threads $cpu_threads -cpu_iterations $cpu_iterations
    ./build/benchmark -queues BrokerQueue -cpu_threads $cpu_threads -cpu_iterations $cpu_iterations >> $RESULT_FILE
done



