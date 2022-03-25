#!/usr/bin/env python

"""
  validate.py
"""

import numpy as np
import argparse

# --
# Helpers

def compute_recall(targets, scores):
    assert targets.shape == scores.shape
    
    recalls = []
    for t, p in zip(targets, scores):
        recalls.append(len(set.intersection(set(t), set(p))) / len(p))
        
    return np.mean(recalls)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
  args      = parse_args()
  topk_path = args.inpath.replace('.cache.bin', '.cache.topk.npy')
  
  act    = np.load(topk_path).astype(np.int64)
  pred   = np.loadtxt('results/scores', dtype=np.int64)
  act    = act[:,:pred.shape[1]]
  recall = compute_recall(act, pred)
  print(f'recall={recall}')