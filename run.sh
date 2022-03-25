#!/bin/bash

# run.sh

# --
# Build

make clean
make all -j12

# --
# Generate dataset

DB_SIZE=100000
DIM=128

python prob2bin.py --db_size $DB_SIZE --dim $DIM

# Build index
./main \
    --mode           database     \
    --database       data/${DB_SIZE}-${DIM}.bin \
    --databaseSize   $DB_SIZE     \
    --dimension      $DIM         \
    --outputGraph    graph.hnsw   \
    --efConstruction 1024         \
    --M 32

# Dump edgelist - HACK!
./main \
    --mode       query        \
    --query      data/out.bin \
    --querySize  1            \
    --dimension  $DIM         \
    --inputGraph graph.hnsw   > edgelist

# generate .pkl files
python prep.py

# generate cache.bin
python prep2bin.py --db_size $DB_SIZE --dim $DIM --n_query 1024

numactl -C 0   -m 0 -- ./ipnsw cache.bin
numactl -C 0-1 -m 0 -- ./ipnsw cache.bin
numactl -C 0-3 -m 0 -- ./ipnsw cache.bin

rm -rf results ; mkdir -p results
numactl -C 0-7 -m 0 -- ./ipnsw cache.bin 128 128
python validate.py