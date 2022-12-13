#!/bin/bash

RESULT_GPU_FILE="results/queues.txt"
echo "" > $RESULT_GPU_FILE

RESULT_CPU_FILE="results/cpu_queues.txt"
# echo "" > $RESULT_CPU_FILE

for (( ii=1; ii<32; ii = ii + 1 ))
do

echo "itreation $ii:"

for (( rbs=8; rbs>=8; rbs = rbs / 2 ))
do

echo "ring buffer size $rbs:"

TOTAL_ADDS=$((1024*1024))
for (( c=1; c<32; c = c * 2 ))
do
    
    gpu_threads=$c
    gpu_blocks=1
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue,CASRingBuffer -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations 
    ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue,CASRingBuffer -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_GPU_FILE
done

for (( c=32; c<=1024; c = c * 2 ))
do
    
    gpu_threads=$c
    gpu_blocks=1
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue,CASRingBuffer,CASRingBufferRequest -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations 
    ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue,CASRingBuffer,CASRingBufferRequest -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_GPU_FILE
done

TOTAL_ADDS=$((1024*1024*256))

for (( c=512; c>=1; c = c / 2 ))
do
    
    gpu_threads=1024
    gpu_blocks=$c
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations
    ./build/benchmark -ring_buffer_size $rbs -queues BrokerQueueFast,OriginalBrokerQueue -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_GPU_FILE
done

TOTAL_ADDS=$((1024*1024))

for (( c=512; c>=1; c = c / 2 ))
do
    
    gpu_threads=1024
    gpu_blocks=$c
    gpu_iterations=$(( TOTAL_ADDS / (gpu_threads * gpu_blocks) ))
    echo ./build/benchmark -ring_buffer_size $rbs -queues CASRingBuffer,CASRingBufferRequest -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations
    ./build/benchmark -ring_buffer_size $rbs -queues CASRingBuffer,CASRingBufferRequest -gpu_threads $gpu_threads -gpu_blocks $gpu_blocks -gpu_iterations $gpu_iterations >> $RESULT_GPU_FILE

done


done

done
