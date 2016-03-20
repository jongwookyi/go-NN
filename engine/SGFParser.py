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


class DebugProcessor:
    def begin_game(self):
        print "DebugProcessor: begin!"

    def end_game(self):
        print "DebugProcessor: end!"

    def process(self, property_name, property_data):
        print "DebugProcessor: %s =" % property_name, property_data


def parse_vertex(s):
    if len(s) == 0:
        return None # pass
    x = ord(s[0]) - ord('a')
    y = ord(s[1]) - ord('a')
    return (x,y)

class PlayingProcessor:
    def __init__(self, N):
        self.board = Board(N)

    def begin_game(self):
        self.board.clear()

    def end_game(self):
        pass

    def play_stone(self, stone, vertex_str):
        vertex = parse_vertex(vertex_str)
        if vertex: # if not pass
            x, y = vertex
            assert self.board.play_stone(x, y, stone)

    def process(self, property_name, property_data):
        if property_name == "B": # black plays
            self.play_stone(Stone.Black, property_data)
        elif property_name == "W": # white plays
            self.play_stone(Stone.White, property_data)
        elif property_name == "AB": # black handicap stones
            for vertex_str in property_data:
                self.play_stone(Stone.Black, vertex_str)
        elif property_name == "AW": # white handicap stones
            for vertex_str in property_data:
                self.play_stone(Stone.White, vertex_str)
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


def test_DebugProcessor():
    processor = DebugProcessor()
    #parse_SGF("../data/KGS/SGFs/kgs-19-2008-10-new/2008-10-12-5.sgf", processor)
    parse_SGF("../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf", processor)

def test_PrintingProcessor():
    #processor = PrintingProcessor(19)
    #parse_SGF("../data/KGS/SGFs/kgs-19-2008-10-new/2008-10-12-5.sgf", processor)
    #parse_SGF("../data/KGS/SGFs/KGS2001/2000-10-10-1.sgf", processor)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-01-17.sgf", processor)
    #parse_SGF("/home/greg/coding/ML/go/NN/data/KGS/SGFs/kgs-19-2008-02-new/2008-02-15-4.sgf", processor)
    processor = PrintingProcessor(9)
    parse_SGF("/home/greg/coding/ML/go/NN/data/CGOS/9x9/SGFs/2015/11/13/2285.sgf", processor)
    parse_SGF("/home/greg/coding/ML/go/NN/data/CGOS/9x9/SGFs/2015/11/13/2412.sgf", processor)

if __name__ == "__main__":
    #test_DebugProcessor()
    test_PrintingProcessor()








