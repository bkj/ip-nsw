#!/usr/bin/env python

"""
  baseline.py
"""

import argparse
import numpy as np
from time import perf_counter as time
import faiss

from validate import compute_recall

# --

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    return parser.parse_args()

args = parse_args()

n_results = 10

data_path    = args.inpath.replace('.cache.bin', '.data.npy')
queries_path = args.inpath.replace('.cache.bin', '.queries.npy')
topk_path    = args.inpath.replace('.cache.bin', '.cache.topk.npy')

data    = np.load(data_path)
queries = np.load(queries_path)
act     = np.load(topk_path)[:,:n_results]

# --
# Run CPU brute-force

findex = faiss.IndexFlatIP(data.shape[1])
findex.add(data)

t       = time()
_, pred = findex.search(queries, n_results)
ms      = int((time() - t) * 1_000_000)

recall = compute_recall(act, pred)
print(f'recall={recall} | n_data={data.shape[0]} | ms={ms} | throughput={1e6 * queries.shape[0] / ms}')

# # --

# res    = faiss.StandardGpuResources()  # use a single GPU
# gindex = faiss.IndexFlatIP(data.shape[1])
# gindex = faiss.index_cpu_to_gpu(res, 0, gindex)
# gindex.add(data)

# t       = time()
# _, pred = gindex.search(queries, n_results)
# ms      = int((time() - t) * 1_000_000)

# recall = compute_recall(act, pred)
# print(f'recall={recall} | ms={ms} | throughput={1e6 * queries.shape[0] / ms}')
