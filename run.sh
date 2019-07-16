#!/bin/bash

# run.sh

# !! Hacked version of ip-nsw code the dumps the edgelist of the
# IP-NSW graph

# --
# Get data

mkdir -p data

source ~/projects/hive/.token.sh
# get token from https://api-token.sdh.cloud

wget --header "Authorization:$TOKEN" https://sdh.cloud/data/ip-nsw/database_music100.bin \
    -O data/database_music100.bin

wget --header "Authorization:$TOKEN" https://sdh.cloud/data/ip-nsw/query_music100.bin \
    -O data/query_music100.bin

# --
# Build

cmake CMakeLists.txt
make -j12

# --
# Build index

./main \
    --mode database \
    --database data/database_music100.bin \
    --databaseSize 1000000 \
    --dimension 100 \
    --outputGraph graph.hnsw \
    --efConstruction 1024 \
    --M 32

# --
# Dump edgelist

./main \
    --mode query \
    --query data/query_music100.bin \
    --querySize 100 \
    --dimension 100 \
    --inputGraph graph.hnsw \
    --topK 10 \
    --efSearch 128 \
    --output result.txt.tmp > edgelist

wc -l edgelist