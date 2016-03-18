#!/usr/bin/python

import numpy as np

class Stone:
    Empty = 0
    Black = 1
    White = 2

def flipped_stone(stone):
    if stone == Stone.Empty: return Stone.Empty
    elif stone == Stone.Black: return Stone.White
    else: return Stone.Black # stone == Stone.White

dxdys = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Board:
    def __init__(self, N):
        self.N = N
        self.clear()

    def clear(self):
        self.vertices = np.empty((self.N, self.N), dtype=np.int32)
        self.vertices.fill(Stone.Empty)
        self.history = []
        self.move_list = []

    def __getitem__(self, index):
        return self.vertices[index]

    def is_on_board(self, x, y):
        return x >= 0 and x < self.N and y >= 0 and y < self.N

    def check_group_liberties(self, start_x, start_y, capture=False):
        group_xys = [(start_x, start_y)]
        visited = np.zeros((self.N, self.N), dtype=np.bool_) 
        visited[start_x, start_y] = True
        group_color = self.vertices[start_x, start_y]

        i = 0
        while i < len(group_xys):
            x,y = group_xys[i]
            i += 1
            for dx,dy in dxdys:
                adj_x, adj_y = x+dx, y+dy
                if self.is_on_board(adj_x, adj_y):
                    adj_stone = self.vertices[adj_x, adj_y]
                    if adj_stone == Stone.Empty:
                        return True
                    elif adj_stone == group_color and not visited[adj_x, adj_y]:
                        group_xys.append((adj_x, adj_y))
                        visited[adj_x, adj_y] = True
        # no liberties
        if capture:
            for x,y in group_xys:
                self.vertices[x,y] = Stone.Empty
        return False


    # returns whether the move was legal
    def play_stone(self, x, y, stone, just_testing=False):
        if self.vertices[x, y] != Stone.Empty: return False
        self.vertices[x, y] = stone

        made_a_capture = False
        for dx,dy in dxdys:
            adj_x, adj_y = x+dx, y+dy
            if self.is_on_board(adj_x, adj_y):
                if not self.check_group_liberties(adj_x, adj_y, capture=True):
                    made_a_capture = True

        # check if the new board state repeats a previous state, which is illegal
        for prev_vertices in self.history:
            if np.array_equal(prev_vertices, self.vertices):
                return False

        if not just_testing:
            self.history.append(np.copy(self.vertices))
            self.move_list.append((x,y))

        return made_a_capture or self.check_group_liberties(x, y)

    def save(self):
        self.saved_vertices = np.copy(self.vertices)

    def restore(self):
        self.vertices = saved_vertices

    def play_is_legal(self, x, y, stone):
        self.save()
        legal = self.play_stone(x, y, stone, just_testing=True)
        self.restore()
        return legal

    def flip_colors(self):
        for x in range(self.N):
            for y in range(self.N):
                self.vertices[x,y] = flipped_stone(self.vertices[x,y])

    def show(self):
        stone_strings = { 
                Stone.Empty: '.',
                Stone.Black: '\033[31m0\033[0m',
                Stone.White: '\033[37m0\033[0m' }
        for x in range(self.N): print "=",
        print
        for y in range(self.N):
            for x in range(self.N):
                print stone_strings[self.vertices[x,y]],
            print
        for x in range(self.N): print "=",
        print


def show_sequence(board, moves, first_color):
    board.clear()
    color = first_color
    for x,y in moves:
        legal = board.play_stone(x, y, color)
        board.show()
        print "move was legal?", legal
        color = flipped_stone(color)


def test_Board():
    board = Board(5)

    print "simplest capture:"
    show_sequence(board, [(1, 0), (0, 0), (0, 1)], Stone.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Stone.White)
    board.flip_colors()

    print "bigger capture:"
    show_sequence(board, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)], Stone.Black)

    print "ko:"
    show_sequence(board, [(0, 1), (3, 1), (1, 0), (2, 0), (1, 2), (2, 2), (2, 1), (1, 1)], Stone.Black)
    print "move at (2, 1) is legal?", board.play_is_legal(2, 1, Stone.Black)
    board.show()
    board.flip_colors()
    print "fipped board:"
    board.show()

    print "self capture:"
    show_sequence(board, [(0, 1), (1, 1), (1, 0)], Stone.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Stone.White)




#test_Board()
