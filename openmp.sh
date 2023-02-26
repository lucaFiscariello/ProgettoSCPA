#!/bin/bash 

for (( j=0; j<48; j++ ));
do
    export OMP_NUM_THREADS=$j
    ./bin/release
    mv results_123456789.csv $j.csv
done


