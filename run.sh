#!/bin/bash

# run.sh

# --
# Build

mkdir -p data
make clean
make all -j12

# Generate dataset

DIM=128
for DB_SIZE in $(cat db_sizes.txt); do
    python prob2bin.py --seed 234234 --db_size $DB_SIZE --dim $DIM --inpath /home/ben/projects/redem/data/ann
done

for DB_SIZE in $(cat db_sizes.txt); do
    # Build index
    ./main \
        --mode           database                      \
        --database       data/${DB_SIZE}-${DIM}.bin    \
        --databaseSize   $DB_SIZE                      \
        --dimension      $DIM                          \
        --outputGraph    data/${DB_SIZE}-${DIM}.hnsw   \
        --efConstruction 1024                          \
        --M 32

    # Dump edgelist - HACK!
    ./main \
        --mode       query        \
        --query      data/${DB_SIZE}-${DIM}.bin \
        --querySize  1            \
        --dimension  $DIM         \
        --inputGraph data/${DB_SIZE}-${DIM}.hnsw > data/${DB_SIZE}-${DIM}.edgelist
    
    # generate .pkl files
    python prep.py --inpath data/${DB_SIZE}-${DIM}.edgelist
    
    # generate cache.bin
    python prep2bin.py --db_size $DB_SIZE --dim $DIM --n_query 1024
done

# --
# Run experiments

rm -rf results ; mkdir -p results

for PROB in $(find data -type f | fgrep .cache.bin | sort -n); do

find data -type f | fgrep .cache.bin | sort -n | xargs -I {} numactl -C 0    -m 0 -- ./ipnsw {} 10
find data -type f | fgrep .cache.bin | sort -n | xargs -I {} numactl -C 0-7  -m 0 -- ./ipnsw {} 10
find data -type f | fgrep .cache.bin | sort -n | xargs -I {} numactl -C 0-23 -m 0 -- ./ipnsw {} 10
find data -type f | fgrep .cache.bin | sort -n | xargs -I {} numactl -C 0-23 -m 0 -- python baseline.py --inpath {}

# IPNSW
for PROB in $(find data -type f | fgrep .cache.bin | sort -n); do
    # my ipnsw
    numactl -C 0   -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-1 -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-3 -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-7 -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-23 -m 0 -- ./ipnsw $PROB 10
    # python validate.py --inpath $PROB
    
    # my baseline
    # numactl -C 0-7 -m 0 -- ./brute_force $PROB
    # python validate.py --inpath $PROB
    
    # FAISS baseline
    # numactl -C 0-23 -m 0 -- python baseline.py --inpath $PROB
done

# complexity of brute force is linear w/ the size of the DB
# complexity of IPNSW is logarithmic w/ the size of the DB

# Consequentially:
#   - If you shard the index, you get n_shard * log(db_size / n_shard) = n_shard * log(db_size) - log(n_shard).  No good.
#   - You could brute force search ... if you could fit it all in GPU memory. It'd take ~ 480 seconds.  Not terrible.  Not great either.