#!/bin/bash

# run.sh

# --
# Prep problem

# python prob2bin.py

# --
# Run

rm -rf results
mkdir -p results

make clean
make -j12

for NUM_THREADS in 1 2 4 8 10 20 40; do
    OMP_NUM_THREADS=$NUM_THREADS ./ipnsw cache.bin 10
    cat results/elapsed
done

python ../validate.py --ref-dir ../data --res-dir results