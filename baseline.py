import numpy as np
from time import perf_counter as time
import faiss

from validate import compute_recall

# --

n_results = 10

data    = np.load('data.npy')
queries = np.load('queries.npy')
act     = np.load('cache.topk.npy')[:,:n_results]

# --
# Run

findex = faiss.IndexFlatIP(data.shape[1])
findex.add(data)

t       = time()
_, pred = findex.search(queries, n_results)
ms      = int((time() - t) * 1_000_000)

recall = compute_recall(act, pred)
print(f'recall={recall} | ms={elapsed}')
