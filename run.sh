#!/bin/bash

# run.sh

# --
# Build

make clean
make all -j12

# Generate dataset

DIM=128
for DB_SIZE in $(cat db_sizes.txt); do
    python prob2bin.py --db_size $DB_SIZE --dim $DIM
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
        --query      data/out.bin \
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

# IPNSW
for PROB in $(find data -type f | fgrep .cache.bin | sort -n); do
    echo $PROB
    
    # my ipnsw
    # numactl -C 0   -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-1 -m 0 -- ./ipnsw $PROB 10
    # numactl -C 0-3 -m 0 -- ./ipnsw $PROB 10
    numactl -C 0-7 -m 0 -- ./ipnsw $PROB 10
    # python validate.py --inpath $PROB
    
    # my baseline
    # numactl -C 0-7 -m 0 -- ./brute_force $PROB
    # python validate.py --inpath $PROB
    
    # FAISS baseline
    python baseline.py --inpath $PROB
done

# complexity of brute force is linear w/ the size of the DB
# complexity of IPNSW is logarithmic w/ the size of the DB

# Consequentially:
#   - If you shard the index, you get n_shard * log(db_size / n_shard) = n_shard * log(db_size) - log(n_shard).  No good.
#   - 
#   - You could brute force search ... if you could fit it all in GPU memory. It'd take ~ 480 seconds.  Not terrible.  Not great either.