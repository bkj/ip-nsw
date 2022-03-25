#!/usr/bin/env python

"""
  prob2bin.py
"""

import os
import sys
import bcolz
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str, default='/home/ubuntu/projects/redem/data/ann/v0/')
    parser.add_argument('--db_size', type=int, default=10_000)
    parser.add_argument('--dim',     type=int, default=256)
    parser.add_argument('--seed',    type=int, default=123)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# --
# IO

x       = bcolz.carray(rootdir=os.path.join(args.inpath, 'all.bcolz'), mode='r')

# --
# Sample

sel     = np.random.choice(x.shape[0], args.db_size, replace=False)
sel     = np.sort(sel)
x       = x[sel]
x       = x[:,:args.dim]

print(f'x.shape={x.shape}', file=sys.stderr)

# !! should also sample queries

# --
# Save

# binary
binpath = f'data/{args.db_size}-{args.dim}.bin'
print(f'writing to: {binpath}', file=sys.stderr)
with open(binpath, 'wb') as f:
  _ = f.write(bytearray(x.ravel()))

# numpy
npypath = binpath.replace('.bin', '.npy')
print(f'writing to: {npypath}', file=sys.stderr)
np.save(npypath, x)
