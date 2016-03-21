from Board import Board

class BaseEngine(object):
    def __init__(self):
        self.board = None

    # subclasses must override this
    def name(self):
        assert False

    # subclasses must override this
    def version(self):
        assert False

    # subclasses may override to only accept
    # certain board sizes. They should call this
    # base method.
    def set_board_size(self, N):
        self.board = Board(N)
        return True

    def clear_board(self):
        self.board.clear()
        self.opponent_passed = False

    def set_komi(self, komi):
        pass

    def player_passed(self, color):
        self.opponent_passed = True

    def stone_played(self, x, y, color):
        assert self.board.play_stone(x, y, color)
        self.opponent_passed = False
        self.board.show()

    # subclasses must override this
    def pick_move(self, color):
        assert False

    def generate_move(self, color):
        coords = self.pick_move(color)
        if coords:
            x,y = coords
            assert self.board.play_stone(x, y, color)
        self.board.show()
        return coords

    def quit(self):
        pass


class IdiotEngine(BaseEngine):
    def __init__(self):
        super(IdiotEngine,self).__init__() 

    def name(self):
        return "IdiotEngine"

    def version(self):
        return "1.0"

    def pick_move(self, color):
        for x in xrange(self.board.N):
            for y in xrange(self.board.N):
                if self.board.play_is_legal(x, y, color):
                    return (x,y)
        return None


