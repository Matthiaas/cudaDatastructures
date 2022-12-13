#!/bin/bash

RESULT_GPU_FILE="results/queues.txt"
echo "" > $RESULT_GPU_FILE

RESULT_CPU_FILE="results/cpu_queues.txt"
# echo "" > $RESULT_CPU_FILE

for (( ii=1; ii<32; ii = ii + 1 ))
do

echo "itreation $ii:"

for (( rbs=256; rbs>=1; rbs = rbs / 2 ))
do

echo "ring buffer size $rbs:"


TOTAL_ADDS=$((1024*1024*32))


for (( c=32; c>=1; c = c / 2 ))
do
    
    cpu_threads=$c
    cpu_blocks=1
    cpu_iterations=$(( TOTAL_ADDS / (cpu_threads * cpu_blocks) ))
    echo ./build/benchmark -ring_buffer_size $rbs -queues LockRingBuffer,CASRingBuffer,BrokerQueue -cpu_threads $cpu_threads -cpu_iterations $cpu_iterations
    ./build/benchmark -ring_buffer_size $rbs -queues LockRingBuffer,CASRingBuffer,BrokerQueue -cpu_threads $cpu_threads -cpu_iterations $cpu_iterations >> $RESULT_CPU_FILE
done


done

done
