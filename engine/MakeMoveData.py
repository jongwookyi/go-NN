#!/usr/bin/python

import numpy as np
import struct
import sys
import os
import os.path
import time

from SGFParser import *
from Board import *
import Features

def make_move_arr(x, y):
    return np.array([x,y], dtype=np.int8)

def show_plane(array):
    assert len(array.shape) == 2
    N = array.shape[0]
    print "=" * N
    for y in xrange(N):
        for x in xrange(N):
            sys.stdout.write('1' if array[x,y]==1 else '0')
        sys.stdout.write('\n')
    print "=" * array.shape[1]

def show_all_planes(array):
    assert len(array.shape) == 3
    for i in xrange(array.shape[2]):
        print "PLANE %d:" % i
        show_plane(array[:,:,i])

def show_feature_planes_and_move(feature_planes, move):
    print "FEATURE PLANES:"
    show_all_planes(feature_planes)
    print "MOVE:"
    print move

def show_batch(all_feature_planes, all_moves):
    batch_size = all_feature_planes.shape[0]
    print "MINIBATCH OF SIZE", batch_size
    for i in xrange(batch_size):
        print "EXAMPLE", i
        show_feature_planes_and_move(all_feature_planes[i,:,:,:], all_moves[i,:])

def test_feature_planes():
    board = Board(5)
    moves = [(0,0), (1,1), (2,2), (3,3), (4,4)]
    play_color = Color.Black
    for x,y in moves:
        board.show()
        feature_planes = make_feature_planes(board, play_color)
        move_arr = make_move_arr(x, y)
        show_feature_planes_and_move(feature_planes, move_arr)
        print
        board.play_stone(x, y, play_color)
        play_color = flipped_color[play_color]

def test_minibatch_read_write():
    N = 5
    board = Board(N)
    moves = [(0, 0), (1, 1), (2, 2)]
    minibatch_size = len(moves)
    num_feature_planes = 3
    all_feature_planes = np.zeros((minibatch_size, N, N, num_feature_planes), dtype=np.int8)
    all_moves = np.zeros((minibatch_size, 2), dtype=np.int8)
    play_color = Color.Black
    for i in xrange(minibatch_size):
        print "out example %d" % i
        x,y = moves[i]
        board.show()
        all_feature_planes[i,:,:,:] = make_feature_planes(board, play_color)
        all_moves[i,:] = make_move_arr(x, y)
        show_feature_planes_and_move(all_feature_planes[i,:,:,:], all_moves[i,:])
        print
        board.play_stone(x, y, play_color)
        play_color = flipped_stone(play_color)

    filename = "/tmp/test_minibatch.npz"
    print "writing minibatch..."
    write_minibatch(filename, all_feature_planes, all_moves)
    print "reading minibatch..."
    write_minibatch(filename, all_feature_planes, all_moves)
    (in_feature_planes, in_moves) = read_minibatch(filename)

    for i in xrange(minibatch_size):
        print "in example %d" % i
        show_feature_planes_and_move(in_feature_planes[i,:,:,:], in_moves[i,:])


def write_game_data(sgf, writer, feature_maker, rank_allowed):
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        print "skipping game b/c of disallowed rank. ranks are %s, %s" % (reader.black_rank, reader.white_rank)
        return

    while reader.has_more():
        vertex, color = reader.peek_next_move()
        if vertex: # if now pass:
            feature_planes = feature_maker(reader.board, reader.next_play_color())
            x, y = vertex
            move_arr = make_move_arr(x, y)
            writer.push_example((feature_planes, move_arr))
        reader.play_next_move()

def make_move_prediction_data(sgf_list, N, Nfeat, out_dir, feature_maker):
    sgf_list = list(sgf_list) # make local copy
    random.shuffle(sgf_list)

    writer = RandomizingNpzWriter(out_dir=out_dir,
            names=['feature_planes', 'moves'],
            shapes=[(N,N,Nfeat), (N,N)],
            dtypes=[np.int8, np.int8],
            Nperfile=128, buffer_len=50000)

    num_games = 0
    for sgf in sgf_list:
        write_game_data(sgf, writer, feature_maker, rank_allowed)
        num_games += 1
        if num_games % 100 == 0: print "num_games =", num_games

    writer.drain()

def make_KGS_sets():
    # TODO
    pass

class TrainingDataWriter:
    def __init__(self, N, out_dir, minibatch_size, feature_maker, Nfeat, rank_allowed):
        self.out_dir = out_dir
        self.player = PlayingProcessor(N)
        self.minibatch_features = np.empty((minibatch_size, N, N, Nfeat), dtype=np.int8)
        self.minibatch_moves = np.empty((minibatch_size, 2), dtype=np.int8)
        self.example_index = 0
        self.minibatch_number = 0
        self.minibatch_size = minibatch_size
        self.feature_maker = feature_maker
        self.Nfeat = Nfeat
        self.rank_allowed = rank_allowed
        self.start_time = time.time()
        self.num_positions = 0

    def begin_game(self):
        self.player.begin_game()
        self.ignore_game = False
    
    def end_game(self):
        self.player.end_game()

    def write_move(self, play_color, move_str):
        vertex = parse_vertex(move_str)
        if not vertex: return # player passed
        x,y = vertex
        self.minibatch_features[self.example_index,:,:,:] = self.feature_maker(self.player.board, play_color)
        self.minibatch_moves[self.example_index,:] = make_move_arr(x, y)
        self.example_index += 1
        if self.example_index == self.minibatch_size:
            filename = "%s/minibatch.%d" % (self.out_dir, self.minibatch_number)
            write_minibatch(filename, self.minibatch_features, self.minibatch_moves)
            self.example_index = 0
            self.minibatch_number += 1
        self.num_positions += 1
        if self.num_positions % 1000 == 0:
            print "positions per second = ", self.num_positions/(time.time() - self.start_time)

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

def test_TrainingDataWrite():
    N = 19
    sgf = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2009-09-new/2009-09-14-52.sgf"
    print "going to read the game at", sgf

    print "let's play through the game first:"
    parse_SGF(sgf, PrintingProcessor(N))

    print "OK, now let's process it into minibatches:"
    out_dir = "/tmp/train_data_test"
    minibatch_size = 10
    num_features = 3
    writer = TrainingDataWriter(N, out_dir, minibatch_size, num_features)
    parse_SGF(sgf, writer)

    print "Now we will try to read the written minibatch files:"
    for mbfile in os.listdir(out_dir):
        filename = os.path.join(out_dir, mbfile)
        print "filename =", filename
        mb_features, mb_moves = read_minibatch(filename)
        show_batch(mb_features, mb_moves)

def make_KGS_training_data():
    N = 19
    minibatch_size = 128
    Nfeat = 15
    feature_maker = Features.make_feature_planes_stones_3liberties_4history_ko
    dir_name = "stones_3lib_4hist_ko_Nf15"
    out_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/%s" % dir_name
    rank_allowed = lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d',
                                         '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p']
    writer = TrainingDataWriter(N, out_dir, minibatch_size, feature_maker, Nfeat, rank_allowed)
    base_dir = "/home/greg/coding/ML/go/NN/data/KGS/SGFs"
    num_games = 0
    for period_dir in os.listdir(base_dir):
        for sgf_file in os.listdir(os.path.join(base_dir, period_dir)):
            filename = os.path.join(base_dir, period_dir, sgf_file)
            parse_SGF(filename, writer)
            num_games += 1
            if num_games % 100 == 0: print "num_games =", num_games

def make_CGOS9x9_training_data():
    N = 9
    minibatch_size = 1000
    num_features = 24
    out_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d" % (minibatch_size, num_features)
    rank_allowed = lambda rank: True
    writer = TrainingDataWriter(N, out_dir, minibatch_size, num_features, rank_allowed)
    base_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/SGFs"
    num_games = 0
    for year_dir in os.listdir(base_dir):
        for month_dir in os.listdir(os.path.join(base_dir, year_dir)):
            for day_dir in os.listdir(os.path.join(base_dir, year_dir, month_dir)):
                for sgf_file in os.listdir(os.path.join(base_dir, year_dir, month_dir, day_dir)):
                    filename = os.path.join(base_dir, year_dir, month_dir, day_dir, sgf_file)
                    parse_SGF(filename, writer)
                    num_games += 1
                    if num_games % 100 == 0: print "num_games =", num_games
        

if __name__ == "__main__":
    #test_feature_planes()
    #test_minibatch_read_write()
    #test_TrainingDataWrite()
    #run_PlaneTester()
    
    make_KGS_training_data()
    #make_CGOS9x9_training_data()
    
    #import cProfile
    #cProfile.run('make_KGS_training_data()', sort='cumtime')

