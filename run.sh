#!/bin/bash

# run.sh

# --
# Get data

mkdir -p data

source ~/projects/hive/.token.sh

wget --header "Authorization:$TOKEN" https://sdh.cloud/data/ip-nsw/database_music100.bin \
    -O data/database_music100.bin

wget --header "Authorization:$TOKEN" https://sdh.cloud/data/ip-nsw/query_music100.bin \
    -O data/query_music100.bin

# --
# Build

cmake CMakeLists.txt
make -j12

# --
# Index

./main \
    --mode database \
    --database data/database_music100.bin \
    --databaseSize 10000 \
    --dimension 100 \
    --outputGraph out_graph.hnsw.tmp \
    --efConstruction 1024 \
    --M 32

# --
# Run query

./main \
    --mode query \
    --query data/query_music100.bin \
    --querySize 100 \
    --dimension 100 \
    --inputGraph out_graph.hnsw.tmp \
    --topK 10 \
    --efSearch 128 \
    --output result.txt.tmp > edges.tmp