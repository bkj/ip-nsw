#!/usr/bin/env python

"""
    ip-nsw/prep.py
"""

import sys
import pickle
import argparse
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str)
    args = parser.parse_args()
    
    args.outpath = args.inpath.replace('.edgelist', '.pkl')
    
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    # --
    # IO
    
    print('prep.py: reading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath, sep=' ', names=['level', 'src', 'dst'])
    print('prep.py: read %d edges' % edges.shape[0], file=sys.stderr)
    
    print(set(edges.level))
    assert len(set(edges.level)) == 4
    
    # --
    # Make graphs
    
    graphs = {}
    
    for level in [0, 1, 2, 3]:
        graphs[level] = {}
        
        tmp = edges[edges.level == level]
        tmp = tmp[['src', 'dst']].values
        
        start_idx = 0
        last_src  = tmp[0,0]
        for idx, (src, _) in tqdm(enumerate(tmp), total=tmp.shape[0]):
            if src != last_src:
                graphs[level][last_src] = tmp[start_idx:idx,1]
                start_idx = idx
                last_src  = src
        
        graphs[level][last_src] = tmp[start_idx:,1]
    
    print('prep.py: writing %s' % args.outpath, file=sys.stderr)
    _ = pickle.dump(graphs, open(args.outpath, 'wb'))