#!/usr/bin/env python

"""
    ipnsw/main.py
    
    Note to program performers:
        - Results should match those from program IP-NSW workflow (#16 in December release)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from time import time
import faiss

from scipy import sparse

# --
# Helpers

def dict2csr(G, n):    
  nodes = sorted(list(G.keys()))
  
  # assert min(nodes) == 0
  # assert max(nodes) == len(nodes) - 1
  
  row = np.concatenate([[node] * len(G[node]) for node in nodes])
  col = np.concatenate([G[node] for node in nodes])
  val = np.ones(len(row)).astype(np.float32)
  
  csr = sparse.csr_matrix((val, (row, col)), shape=(n, n)).tocsr()
  return csr

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_size',  type=int, required=True)
    parser.add_argument('--dim',      type=int, required=True)
    parser.add_argument('--n_query',  type=int, required=True)
    return parser.parse_args()

args = parse_args()

print('reading', file=sys.stderr)
graphs  = pickle.load(open(f'./data/{args.db_size}-{args.dim}.pkl', 'rb'))
data    = np.fromfile(f'./data/{args.db_size}-{args.dim}.bin', dtype=np.float32).reshape(args.db_size, args.dim)
queries = data[np.random.choice(data.shape[0], args.n_query, replace=False)]

# >>
print('compute ground truth')
findex = faiss.IndexFlatIP(data.shape[1])
findex.add(data)
_, topk = findex.search(queries, 1024)

np.save(f'data/{args.db_size}-{args.dim}.data.npy', data)
np.save(f'data/{args.db_size}-{args.dim}.queries.npy', queries)
np.save(f'data/{args.db_size}-{args.dim}.cache.topk.npy', topk)
# <<

print('converting', file=sys.stderr)
G0 = dict2csr(graphs[0], n=args.db_size)
G1 = dict2csr(graphs[1], n=args.db_size)
G2 = dict2csr(graphs[2], n=args.db_size)
G3 = dict2csr(graphs[3], n=args.db_size)

data_shape    = np.array(data.shape).astype(np.int32)
queries_shape = np.array(queries.shape).astype(np.int32)

G0_shape = np.array(G0.shape).astype(np.int32)
G0_nnz   = np.array([G0.nnz]).astype(np.int32)

G1_shape = np.array(G1.shape).astype(np.int32)
G1_nnz   = np.array([G1.nnz]).astype(np.int32)

G2_shape = np.array(G2.shape).astype(np.int32)
G2_nnz   = np.array([G2.nnz]).astype(np.int32)

G3_shape = np.array(G3.shape).astype(np.int32)
G3_nnz   = np.array([G3.nnz]).astype(np.int32)

data    = data.astype(np.float32)
queries = queries.astype(np.float32)

print(f'writing data/{args.db_size}-{args.dim}.cache.bin')
with open(f'data/{args.db_size}-{args.dim}.cache.bin', 'wb') as f:
  _ = f.write(bytearray(data_shape))
  _ = f.write(bytearray(data.ravel()))

  _ = f.write(bytearray(queries_shape))
  _ = f.write(bytearray(queries.ravel()))

  _ = f.write(bytearray(G0_shape))
  _ = f.write(bytearray(G0_nnz))
  _ = f.write(bytearray(G0.indptr))
  _ = f.write(bytearray(G0.indices))
  _ = f.write(bytearray(G0.data))

  _ = f.write(bytearray(G1_shape))
  _ = f.write(bytearray(G1_nnz))
  _ = f.write(bytearray(G1.indptr))
  _ = f.write(bytearray(G1.indices))
  _ = f.write(bytearray(G1.data))

  _ = f.write(bytearray(G2_shape))
  _ = f.write(bytearray(G2_nnz))
  _ = f.write(bytearray(G2.indptr))
  _ = f.write(bytearray(G2.indices))
  _ = f.write(bytearray(G2.data))

  _ = f.write(bytearray(G3_shape))
  _ = f.write(bytearray(G3_nnz))
  _ = f.write(bytearray(G3.indptr))
  _ = f.write(bytearray(G3.indices))
  _ = f.write(bytearray(G3.data))