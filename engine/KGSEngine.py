from Engine import *
from HelperEngine import HelperEngine

# forwards commands to both a main engine
# and a helper engine. When picking a move,
# we first ask the helper engine. If it passes,
# we pass. Otherwise we ask the main engine
class KGSEngine(BaseEngine):
    def __init__(self, engine):
        self.engine = engine
        self.helper_pass = HelperEngine(cleanup=False)
        self.helper_cleanup = HelperEngine(cleanup=True)

    # subclasses must override this
    def name(self):
        return self.engine.name()

    # subclasses must override this
    def version(self):
        return self.engine.version()

    def set_board_size(self, N):
        return self.engine.set_board_size(N) and \
               self.helper_pass.set_board_size(N) and \
               self.helper_cleanup.set_board_size(N)

    def clear_board(self):
        self.engine.clear_board()
        self.helper_pass.clear_board()
        self.helper_cleanup.clear_board()

    def set_komi(self, komi):
        self.engine.set_komi(komi)
        self.helper_pass.set_komi(komi)
        self.helper_cleanup.set_komi(komi)

    def player_passed(self, color):
        self.engine.player_passed(color)
        self.helper_pass.player_passed(color)
        self.helper_cleanup.player_passed(color)

    def stone_played(self, x, y, color):
        self.engine.stone_played(x, y, color)
        self.helper_pass.stone_played(x, y, color)
        self.helper_cleanup.stone_played(x, y, color)

    def generate_move(self, color, cleanup=False):
        if cleanup:
            move = self.helper_cleanup.generate_move(color)
            if move.is_play():
                self.engine.stone_played(move.x, move.y, color)
                self.helper_pass.stone_played(move.x, move.y, color)
            elif move.is_pass():
                self.engine.player_passed(color)
                self.helper_pass.player_passed(color)
            return move

        helper_move = self.helper_pass.generate_move(color)
        if helper_move.is_pass(): # helper passed
            self.engine.player_passed(color)
            self.helper_cleanup.player_passed(color)
            return Move.Pass
        elif helper_move.is_resign(): # helper resigned
            return Move.Resign
        else: # helper didn't pass or resign
            if helper_move.is_play():
                self.helper_pass.undo() # helper must support this
            move = self.engine.generate_move(color)
            if move.is_play(): 
                self.helper_pass.stone_played(move.x, move.y, color)
                self.helper_cleanup.stone_played(move.x, move.y, color)
            elif move.is_pass(): 
                self.helper_pass.player_passed(color)
                self.helper_cleanup.player_passed(color)
            return move

    def undo(self):
        self.engine.undo()
        self.helper_pass.undo()
        self.helper_cleanup.undo()

    def quit(self):
        self.engine.quit()
        self.helper_pass.quit()
        self.helper_cleanup.quit()

    def supports_final_status_list(self):
        return True

    def final_status_list(self, status):
        return self.helper_cleanup.final_status_list(status)
