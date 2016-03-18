#!/usr/bin/python

import numpy as np
import struct

from SGFParser import *
from Board import *

def write_image_planes(f, array):
    assert array.dtype == np.int32
    format_str = 'b' * array.size
    f.write(struct.pack(format_str, *array.flat))

def read_image_planes(f, array):
    assert array.dtype == np.int32
    format_str = 'b' * array.size
    flat_data = struct.unpack(format_str, f.read(array.size))
    np.copyto(array, np.array(flat_data).reshape(array.shape))


def make_stone_plane(array, board, stone):
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board[x,y] == stone: array[x,y] = 1

def make_ones_plane(array, board):
    np.copyto(array, np.ones((board.N, board.N)))

def make_history_planes(array, board, max_lookback):
    assert array.shape[2] == max_lookback
    for lookback in xrange(max_lookback):
        if lookback < len(board.move_history):
            x,y = board.move_history[lookback]
            array[x,y,lookback] = 1

def find_group(board, start_x, start_y):
    group_xys = [(start_x, start_y)]
    visited = np.zeros((board.N, board.N), dtype=np.bool_) 
    visited[start_x, start_y] = True
    group_color = board[start_x, start_y]
    i = 0
    while i < len(group_xys):
        x,y = group_xys[i]
        i += 1
        for dx,dy in dxdys:
            adj_x, adj_y = x+dx, y+dy
            if board.is_on_board(adj_x, adj_y):
                adj_stone = board[adj_x, adj_y]
                if adj_stone == group_color and not visited[adj_x, adj_y]:
                    group_xys.append((adj_x, adj_y))
                    visited[adj_x, adj_y] = True
    return group_xys

def count_group_liberties(board, start_x, start_y):
    group_xys = [(start_x, start_y)]
    visited = np.zeros((board.N, board.N), dtype=np.bool_) 
    visited[start_x, start_y] = True
    group_color = board[start_x, start_y]
    liberties = set()
    i = 0
    while i < len(group_xys):
        x,y = group_xys[i]
        i += 1
        for dx,dy in dxdys:
            adj_x, adj_y = x+dx, y+dy
            if board.is_on_board(adj_x, adj_y):
                adj_stone = board[adj_x, adj_y]
                if adj_stone == Stone.Empty:
                    liberties.add((adj_x, adj_y))
                elif adj_stone == group_color and not visited[adj_x, adj_y]:
                    group_xys.append((adj_x, adj_y))
                    visited[adj_x, adj_y] = True
    return len(liberties)

def make_liberty_count_planes(array, board, Nplanes):
    assert array.shape[2] == Nplanes
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board[x,y] != Stone.Empty:
                num_liberties = board.count_group_liberties(x, y)
                if num_liberties > Nplanes: num_liberties = Nplanes
                array[x,y,num_libiertes-1] = 1

def make_liberty_count_after_move_planes(array, board, Nplanes, stone):
    assert array.shape[2] == Nplanes
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board[x,y] != Stone.Empty:
                board.save()
                if board.play_stone(x, y, stone):
                    num_liberties = board.count_group_liberties(x, y)
                    if num_liberties > Nplanes: num_liberties = Nplanes
                    array[x,y,num_libiertes-1] = 1
                board.restore()

def make_self_atari_size_planes(array, board, Nplanes, stone):
    assert array.shape[2] == Nplanes
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board[x,y] != Stone.Empty:
                board.save()
                if board.play_stone(x, y, stone):
                    num_liberties = board.count_group_liberties(x, y)
                    if num_liberties == 1:
                        group_size = len(find_group(board, x, y))
                        if group_size > Nplanes: group_size = Nplanes
                        array[x,y,group_size-1] = 1
                board.restore()

def count_stones(board, color):
    return np.count_nonzero(np.equal(board.vertices, color))

def count_captured_stones(board, x, y, stone):
    opponent_color = flipped_stone(stone)
    num_opponent_stones = count_stones(board, opponent_color)
    num_captured = 0
    if board[x, y] == Stone.Empty:
        board.save()
        if board.play_stone(x, y, stone, just_testing=True):
            num_captured = num_opponent_stones - count_stones(board, opponent_color)
        board.restore()
    return num_captured

def make_capture_count_planes(array, board, max_capture_count, play_color):
    assert array.shape[2] == max_capture_count
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board[x,y] == Stone.empty:
                num_captured = count_captured_stones(board, x, y, play_color)
                if num_captured >= max_capture_count: num_captured = max_capture_count-1
                array[x,y,num_captured] = 1

def make_legality_plane(board, array, stone):
    for x in xrange(board.N):
        for y in xrange(board.N):
            if board.play_is_legal(x, y, stone):
                array[x,y] = 1

def make_feature_planes(board, play_color):
    Nplanes = 3
    feature_planes = np.zeros((board.N, board.N, Nplanes))
    make_stone_plane(feature_planes[:,:,0], board, play_color)
    make_stone_plane(feature_planes[:,:,1], board, flipped_stone(play_color))
    make_stone_plane(feature_planes[:,:,2], board, Stone.Empty)
    make_ones_plane(feature_planes[:,:,3], board)
    max_lookback = 8
    make_history_planes(feature_planes[:,:,4:12], board, max_lookback)
    max_liberties = 8
    make_liberty_count_planes(feature_planes[:,:,12:20], board, max_liberties)
    max_captures = 8
    make_capture_count_planes(feature_planes[:,:,20:28], board, max_captures, play_color)
    max_self_atari_size = 8
    make_self_atari_size_planes(feature_planes[:,:,28:36], board, max_self_atari_size, play_color)
    max_liberties_after_move = 8
    make_liberty_count_after_move_planes(feature_planes[:,:,36:44], board, max_liberties_after_move, play_color)
    make_legality_plane(feature_planes[:,:,44], board, play_color)
    return feature_planes

def make_move_plane(board, x, y):
    plane = np.zeros((board.N, board.N))
    plane[x,y] = 1
    return plane


class TrainingDataWriter:
    def __init__(self, N, training_data_file):
        self.training_data_file = training_data_file
        self.player = PlayingProcessor(N)

    def __enter__(self):
        self.fout = open(self.training_data_file, 'w')

    def __exit__(self, exception_type, exception_value, traceback):
        self.fout.close()

    def begin(self):
        self.player.begin()
    
    def end(self):
        self.player.end()

    def write_move(self, play_color, move_str):
        vertex = parse_vertex(move_str)
        if vertex:
            x,y = vertex
            feature_planes = make_feature_planes(self.player.board, play_color)
            move_plane = make_move_plane(self.player.board, x, y)
            write_image_planes(self.fout, feature_planes)
            write_image_planes(self.fout, move_plane)

    def process(self, property_name, property_data):
        if property_name == "W":
            pass
        elif proprty_name == "B"
            pass

        self.player.process()

