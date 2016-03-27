import os
import os.path
import random
import time
import numpy as np
from MakeTrainingData import read_minibatch

class NpzMinibatcher:
    def __init__(self, npz_dir):
        self.filename_queue = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir)]
    def has_more(self):
        return len(self.filename_queue) > 0
    def next_minibatch(self):
        return read_minibatch(self.filename_queue.pop())

class RandomizingNpzMinibatcher:
    def __init__(self, npz_dir):
        self.filename_queue = None
        self.npz_dir = npz_dir
    def next_minibatch(self):
        if not self.filename_queue:
            self.filename_queue = [os.path.join(self.npz_dir, f) for f in os.listdir(self.npz_dir)]
            random.shuffle(self.filename_queue)
            print "RandomizingNpzMinibatcher: built new filename queue with length", len(self.filename_queue)
        return read_minibatch(self.filename_queue.pop())


class GroupingRandomizingNpzMinibatcher:
    def __init__(self, npz_dir, Ngroup):
        self.filename_queue = []
        self.npz_dir = npz_dir
        self.Ngroup = Ngroup
    def next_minibatch(self):
        if len(self.filename_queue) < self.Ngroup:
            self.filename_queue = [os.path.join(self.npz_dir, f) for f in os.listdir(self.npz_dir)]
            random.shuffle(self.filename_queue)
            print "RandomizingNpzMinibatcher: built new filename queue with length", len(self.filename_queue)
        components = [read_minibatch(self.filename_queue.pop()) for i in xrange(self.Ngroup)]
        Nperfile = components[0][0].shape[0]
        N = components[0][0].shape[1]
        Nfeat = components[0][0].shape[3]
        grouped_features = np.empty((Nperfile * self.Ngroup, N, N, Nfeat), dtype=np.int8)
        grouped_moves = np.empty((Nperfile * self.Ngroup, 2), dtype=np.int8)
        for i in xrange(self.Ngroup):
            start = i * Nperfile
            end = (i+1) * Nperfile
            grouped_features[start:end,:,:,:], grouped_moves[start:end,:] = components[i]
        return grouped_features, grouped_moves



"""
class RandomizingNpzMinibatcher:
    def __init__(self, npz_dir, N, Nfeat, minibatch_size):
        self.npz_dir = npz_dir
        self.N = N
        self.Nfeat = Nfeat
        self.minibatch_size = minibatch_size
        self.all_filenames = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir)]
        self.make_filename_queue()
        self.prepared_minibatches = []

    def make_filename_queue(self):
        self.filename_queue = list(self.all_filenames)
        random.shuffle(self.filename_queue)

    def next_filename(self):
        if not self.filename_queue:
            self.make_filename_queue()
        return self.filename_queue.pop()

    def next_minibatch(self):
        if not self.prepared_minibatches:
            Nperfile = 1000
            Nfiles = 256
            assert self.minibatch_size == 128
            Nbatch = 2000
            assert self.minibatch_size * Nbatch == Nperfile * Nfiles
            print "Preparing %d minibatches..." % Nbatch
            start_time = time.time()
            prepared_features = np.empty((Nfiles * Nperfile, self.N, self.N, self.Nfeat), dtype=np.int8)
            prepared_moves = np.empty((Nfiles * Nperfile, 2), dtype=np.int8)
            for b in xrange(Nfiles):
                #print "loading", b
                start = b * Nperfile
                end = (b+1) * Nperfile
                features, move = read_minibatch(self.next_filename())
                prepared_features[start:end,:,:,:] = features
                prepared_moves[start:end,:] = move
            print "randomizing..."
            reordering = range(Nfiles * Nperfile)
            random.shuffle(reordering)
            prepared_features = prepared_features[reordering,:,:,:]
            prepared_moves = prepared_moves[reordering,:]
            for b in xrange(Nbatch):
                start = b * self.minibatch_size
                end = (b+1) * self.minibatch_size
                self.prepared_minibatches.append((prepared_features[start:end,:,:,:], prepared_moves[start:end,:]))
            print "Done. Took %.3f seconds." % (time.time() - start_time)

        return self.prepared_minibatches.pop()
"""





"""
class TrainingDataGetter:
    def __init__(self, N, rank_allowed):
        self.player = PlayingProcessor(N)
        self.rank_allowed = rank_allowed
        self.examples = []

    def begin_game(self):
        self.player.begin_game()
        self.ignore_game = False
    
    def end_game(self):
        self.player.end_game()

    def add_example(self, play_color, move_str):
        vertex = parse_vertex(move_str)
        if not vertex: return # play passed
        x,y = vertex
        feature_planes = make_feature_planes(self.player.board, play_color)
        move_arr = make_move_arr(x, y)
        self.examples.append(feature_planes, move_arr)

    def process(self, property_name, property_data):
        if self.ignore_game: return

        if property_name == "W":
            self.write_move(Color.White, property_data)
        elif property_name == "B":
            self.write_move(Color.Black, property_data)
        elif property_name == "WR" or property_name == "BR":
            if not self.rank_allowed(property_data): 
                self.ignore_game = True
                print "ignoring game because it has non-allowed rank:", property_data

        self.player.process(property_name, property_data)


# Parses SGF files as needed
class OnlineMinibatcher:
    def __init__(self, sgf_dir, minibatch_size):
        self.sgf_dir = sgf_dir
        self.N = 19
        self.Nfeat = 24
        self.minibatch_size = minibatch_size
        self.queue = [os.path.join(self.sgf_dir, f) for f in os.listdir(self.sgf_dir)]

        rank_allowed = lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
                                             '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p']
        self.getter = TrainingDataGetter(self.N, rank_allowed)

    def has_more():
        return len(self.queue) > 0

    def next_minibatch(self):
        minibatch_features = np.empty((self.minibatch_size, self.N, self.N, self.Nfeat), dtype=np.int8)
        minibatch_moves = np.empty((self.minibatch_size, 2), dtype=np.int8)
        for i in range(self.minibatch_size):
            while not self.getter.examples == 0:
                parse_SGF(self.queue.pop(), self.getter)
            feature_planes, move_arr = self.getter.examples.pop()
            minibatch_features[i,:,:,:] = feature_planes
            minibatch_moves{i,:,:] = move_arr
        return minibatch_features, minibatch_moves

"""




        




