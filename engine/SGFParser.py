#!/usr/bin/python

from Board import *

READING_NAME = 1
READING_DATA = 2

separators = set(['(', ')', ' ', '\n', '\t', ';'])

properties_taking_lists = set(['AB', # add black stone (handicap)
                               'AW', # add white stone (handicap)
                              ])

def parse_property_name(file_data, ptr):
    while file_data[ptr] in separators: 
        ptr += 1
        if ptr >= len(file_data): return (None, ptr)
    name = ''
    while file_data[ptr] != '[':
        name += file_data[ptr]
        ptr += 1
    return (name, ptr)

def parse_property_data(file_data, ptr):
    while file_data[ptr].isspace(): 
        ptr += 1
    if file_data[ptr] != '[':
        return (None, ptr)
    ptr += 1
    data = ''
    while file_data[ptr] != ']':
        data += file_data[ptr]
        ptr += 1
    ptr += 1
    return (data, ptr)

def parse_property_data_list(file_data, ptr):
    data_list = []
    while True:
        (data, ptr) = parse_property_data(file_data, ptr)
        if data == None:
            return (data_list, ptr)
        else:
            data_list.append(data)

def parse_vertex(s):
    if len(s) == 0:
        return None # pass
    x = ord(s[0]) - ord('a')
    y = ord(s[1]) - ord('a')
    return (x,y)

"""
def parse_SGF(filename, processor):
    print "Parsing SGF file", filename
    with open(filename, 'r') as f:
        file_data = f.read()

    processor.begin_game()
        
    state = READING_NAME

    ptr = 0

    property_name = ""
    property_data = ""

    while True:
        (property_name, ptr) = parse_property_name(file_data, ptr)
        if property_name == None:
            processor.end_game()
            return
        elif property_name in properties_taking_lists:
            (property_data, ptr) = parse_property_data_list(file_data, ptr)
        else:
            (property_data, ptr) = parse_property_data(file_data, ptr)
        processor.process(property_name, property_data)
"""

class SGFParser:
    def __init__(self, filename):
        with open(filename, 'r') as f: 
            self.file_data = f.read()
        self.ptr = 0

    def __iter__(self):
        return self

    def next(self):
        (property_name, self.ptr) = parse_property_name(self.file_data, self.ptr)
        if property_name == None:
            raise StopIteration
        elif property_name in properties_taking_lists:
            (property_data, self.ptr) = parse_property_data_list(self.file_data, self.ptr)
        else:
            (property_data, self.ptr) = parse_property_data(self.file_data, self.ptr)
        return (property_name, property_data)


"""
class DebugProcessor:
    def begin_game(self):
        print "DebugProcessor: begin!"

    def end_game(self):
        print "DebugProcessor: end!"

    def process(self, property_name, property_data):
        print "DebugProcessor: %s =" % property_name, property_data
"""

def test_SGFParser():
    sgf = "../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf"
    parser = SGFParser(sgf)
    for property_name,  property_data in parser:
        print "%s = %s" % (property_name, property_data)


class SGFReader:
    def __init__(self, filename):
        parser = SGFParser(filename)
        self.initial_stones = []
        self.moves = []
        self.black_rank = None
        self.white_rank = None
        for property_name, property_data in parser:
            if property_name == "SZ": # board size
                self.board = Board(int(property_data))
            elif property_name == "AB": # black initial stones
                for vertex_str in property_data:
                    self.initial_stones.append((parse_vertex(vertex_str), Color.Black))
            elif property_name == "AW": # white initial stones
                for vertex_str in property_data:
                    self.initial_stones.append((parse_vertex(vertex_str), Color.White))
            elif property_name == "B": # black plays
                self.moves.append((parse_vertex(property_data), Color.Black))
            elif property_name == "W": # white plays
                self.moves.append((parse_vertex(property_data), Color.White))
            elif property_name == "BR": # black rank
                self.black_rank = property_data
            elif property_name == "WR": # white rank
                self.white_rank = property_data

        for (x,y), color in self.initial_stones:
            self.board.play_stone(x, y, color)

        self.moves_played = 0

    def has_more(self):
        return self.moves_played < len(self.moves)

    def peek_next_move(self):
        return self.moves[self.moves_played]

    def play_next_move(self):
        move = self.moves[self.moves_played]
        self.moves_played += 1
        vertex, color = move
        if vertex:
            x,y = vertex
            self.board.play_stone(x, y, color)
        else:
            self.board.play_pass()
        return move

    def next_play_color(self):
        if self.has_more():
            return self.moves[self.moves_played][1]
        elif self.moves:
            return flipped_color[self.moves[-1][1]]
        elif self.initial_stones:
            return flipped_color[self.initial_stones[-1][1]]
        else:
            return Color.Black


def test_SGFReader():
    sgf = "/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-09-18.sgf"
    reader = SGFReader(sgf)

    print "initial position:"
    reader.board.show()

    while reader.has_more():
        print "before move, next play color is", color_names[reader.next_play_color()]
        vertex, color = reader.play_next_move()
        print "after move", vertex, "by", color_names[color], "board is"
        reader.board.show()
        print "after move, next play color is", color_names[reader.next_play_color()]

    print "Game over."




"""
class PlayingProcessor:
    def __init__(self, N):
        self.board = Board(N)

    def begin_game(self):
        self.board.clear()

    def end_game(self):
        pass

    def play_stone(self, color, vertex_str):
        vertex = parse_vertex(vertex_str)
        if vertex: # if not pass
            x, y = vertex
            self.board.play_stone(x, y, color)
        else:
            self.board.play_pass()

    def process(self, property_name, property_data):
        if property_name == "B": # black plays
            self.play_stone(Color.Black, property_data)
        elif property_name == "W": # white plays
            self.play_stone(Color.White, property_data)
        elif property_name == "AB": # black handicap stones
            for vertex_str in property_data:
                self.play_stone(Color.Black, vertex_str)
        elif property_name == "AW": # white handicap stones
            for vertex_str in property_data:
                self.play_stone(Color.White, vertex_str)
        elif property_name == "SZ": # board size
            assert int(property_data) == self.board.N

class PrintingProcessor:
    def __init__(self, N):
        self.player = PlayingProcessor(N)

    def begin_game(self):
        print "PrintingProcessor: Game start!"
        self.player.begin_game()
        self.move_number = 0

    def end_game(self):
        print "PrintingProcessor: Game end!"
        self.player.end_game()

    properties_causing_prints = set(["B", "W", "AB", "AW"])

    def process(self, property_name, property_data):
        print "PrintingProcessor: %s = " % property_name, property_data
        self.player.process(property_name, property_data)
        if property_name == "W" or property_name == "B":
            self.move_number += 1
            print "Move %d played" % self.move_number
        if property_name in PrintingProcessor.properties_causing_prints:
            print "PrintingProcessor: Now the position is"
            self.player.board.show()
    """




def test_DebugProcessor():
    processor = DebugProcessor()
    #parse_SGF("../data/KGS/SGFs/kgs-19-2008-10-new/2008-10-12-5.sgf", processor)
    parse_SGF("../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf", processor)

def test_PrintingProcessor():
    processor = PrintingProcessor(19)
    #parse_SGF("../data/KGS/SGFs/kgs-19-2008-10-new/2008-10-12-5.sgf", processor)
    #parse_SGF("../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf", processor)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-01-17.sgf", processor)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-15-4.sgf", processor)
    parse_SGF("/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-09-18.sgf", processor)
    #processor = PrintingProcessor(9)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/CGOS/9x9/SGFs/2015/11/13/2285.sgf", processor)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/CGOS/9x9/SGFs/2015/11/13/2412.sgf", processor)

if __name__ == "__main__":
    #test_SGFParser()
    test_SGFReader()








