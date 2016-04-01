#!/usr/bin/python
import os
import pickle
from operator import xor
from collections import defaultdict
from Board import *
from SGFReader import SGFReader

np.random.seed(72936393)
zobrist_keys = np.random.randint(-2**63, 2**63-1, size=(19,19,3))
np.random.seed()

def key_from_board(board):
    return reduce(xor, (zobrist_keys[x,y,board[x,y]] for x in xrange(board.N) for y in xrange(board.N)))

class MoveRecord:
    def __init__(self):
        self.wins = 0
        self.losses = 0

class PositionRecord:
    def __init__(self):
        self.moves = defaultdict(MoveRecord)

def add_game_to_book(sgf, book, max_moves, rank_allowed):
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        print "skipping %s because of invalid rank(s) (%s and %s)" % (sgf, reader.black_rank, reader.white_rank)
        return

    if "B+" in reader.result:
        winner = Color.Black
    elif "W+" in reader.result:
        winner = Color.White
    else:
        print "skipping %s because I can't figure out the winner from \"%s\"" % (sgf, reader.result)
        return

    moves_played = 0
    while moves_played < max_moves and reader.has_more():
        vertex, play_color = reader.peek_next_move()
        if vertex: # if not pass
            board_key = key_from_board(reader.board)
            move_record = book[board_key].moves[vertex]
            if winner == play_color:
                move_record.wins += 1
            else:
                move_record.losses += 1
        reader.play_next_move()
        moves_played += 1

def lookup_position(book, board):
    board_key = key_from_board(board)
    if board_key in book:
        return book[board_key]
    else:
        return None

def make_book_from_GoGoD():
    book = defaultdict(PositionRecord)
    max_moves = 20
    rank_allowed = lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d']
    
    num_games = 0
    top_dir = '/home/greg/coding/ML/go/NN/data/GoGoD/modern_games'
    for sub_dir in os.listdir(top_dir):
        for fn in os.listdir(os.path.join(top_dir, sub_dir)):
            sgf = os.path.join(top_dir, sub_dir, fn)
            print "reading sgf %s" % sgf
            add_game_to_book(sgf, book, max_moves, rank_allowed)
            num_games += 1
            if num_games >= 10000:
                return book

def write_GoGoD_book():
    book = make_book_from_GoGoD()
    print "Writing GoGoD book with %d positions." % len(book)
    with open('/home/greg/coding/ML/go/NN/engine/GoGoDBook.bin', 'w') as f:
        pickle.dump(book, f)

def load_GoGoD_book():
    with open('/home/greg/coding/ML/go/NN/engine/GoGoDBook.bin', 'r') as f:
        book = pickle.load(f)
    print "Loaded GoGoD book with %d positions." % len(book)
    return book

def test_book(book):
    board = Board(19)
    play_color = Color.Black

    for i in xrange(20):
        board.show()
        pos_record = lookup_position(book, board)
        print "known moves:"
        best_vertex = None
        best_count = 0
        for vertex in pos_record.moves:
            move_record = pos_record.moves[vertex]
            print vertex, " - wins=", move_record.wins, "; losses=", move_record.losses
            count = move_record.wins + move_record.losses
            if count > best_count:
                best_count = count
                best_vertex = vertex
        print "best_vertex =", best_vertex
        board.play_stone(best_vertex[0], best_vertex[1], play_color)
        play_color = flipped_color[play_color]


if __name__ == '__main__':
    #test_book()

    write_GoGoD_book()
    book = load_GoGoD_book()
    test_book(book)





    

