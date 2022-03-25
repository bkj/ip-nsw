#!/usr/bin/env python

"""
  validate.py
"""

import numpy as np

# --
# Helpers

def compute_recall(targets, scores):
    assert targets.shape == scores.shape
    
    recalls = []
    for t, p in zip(targets, scores):
        recalls.append(len(set.intersection(set(t), set(p))) / len(p))
        
    return np.mean(recalls)


if __name__ == "__main__":
  act    = np.load('cache.topk.npy').astype(np.int64)
  pred   = np.loadtxt('results/scores', dtype=np.int64)
  act    = act[:,:pred.shape[1]]
  recall = compute_recall(act, pred)
  print(f'recall={recall}')