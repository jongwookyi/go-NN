#!/usr/bin/python
import sys
import os
import os.path
import random
import time
import numpy as np

from MakeTrainingData import read_minibatch, write_minibatch

def randomize_batches(in_dir, out_dir):
    print "in_dir =", in_dir
    print "out_dir =", out_dir
    
    filename_queue = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    random.shuffle(filename_queue)
    total_Nfiles = len(filename_queue)
    
    N = 19
    Nfeat = 24
    
    maxNfiles = 100
    Nperfile = 1000
    
    out_num = 0
    
    start_time = time.time()
    
    while filename_queue:
        Nfiles = min(len(filename_queue), maxNfiles)
        many_features = np.empty((Nfiles * Nperfile, N, N, Nfeat), dtype=np.int8)
        many_moves = np.empty((Nfiles * Nperfile, 2), dtype=np.int8)
    
        for f in xrange(Nfiles):
            start = f * Nperfile
            end = (f+1) * Nperfile
            features, moves = read_minibatch(filename_queue.pop())
            assert Nperfile == features.shape[0] == moves.shape[0]
            many_features[start:end,:,:,:] = features
            many_moves[start:end,:] = moves
        reordering = range(Nfiles * Nperfile)
        random.shuffle(reordering)
        many_features = many_features[reordering,:,:,:]
        many_moves = many_moves[reordering,:]
        for f in xrange(Nfiles):
            start = f * Nperfile
            end = (f+1) * Nperfile
            out_fn = os.path.join(out_dir, 'rand-%d.npz' % out_num)
            out_num += 1
            write_minibatch(out_fn, many_features[start:end,:,:,:], many_moves[start:end,:])
    
        now = time.time()
        time_used = int(now - start_time)
        time_left = len(filename_queue) * time_used / out_num
        print "Finished %d of %d files in %d seconds. Estimate %d seconds remaining." % (out_num, total_Nfiles, time_used, time_left)
        #raw_input()

    print "Done."

dirs = sys.argv[1:]
print "dirs =\n", dirs

assert len(dirs) >= 2

for i in range(len(dirs)-1):
    randomize_batches(dirs[i], dirs[i+1])

