#!/bin/bash

RESULT_FILE="results/hash_table_copy_final.txt"
echo "" > $RESULT_FILE

for (( c=0; c < 32; c = c + 1 ))
do
    
echo ./build/benchmark -hashmapcopy
./build/benchmark -hashmapcopy >> $RESULT_FILE

done