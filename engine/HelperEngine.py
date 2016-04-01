#!/usr/bin/python
import subprocess
import GTP
from Board import *

# Using gnugo to determine when to pass and to play cleanup moves

class HelperEngine:
    def __init__(self):
        # bufsize=1 is line buffered
        self.proc = subprocess.Popen(["gnugo", "--mode", "gtp"], bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send_command(self, command):
        print "HelperEngine: sending command \"%s\"" % command
        self.proc.stdin.write(command)
        self.proc.stdin.write('\n')
    
        response = ""
        while True:
            line = self.proc.stdout.readline()
            if line.startswith('='):
                response += line[2:]
            elif line.startswith('?'):
                print "HelperEngine: error response! line is \"%s\"" % line
                response += line[2:]
            elif len(line.strip()) == 0:
                # blank line ends response
                break
            else:
                response += line
        response = response.strip()
        print "HelperEngine: got response \"%s\"" % response
        return response

    def set_board_size(self, N):
        self.send_command("boardsize %d" % N)
        return True # could parse helper response

    def clear_board(self):
        self.send_command("clear_board")

    def set_komi(self, komi):
        self.send_command("komi %.2f" % komi)

    def player_passed(self, color):
        self.send_command("play %s pass" % color_names[color])

    def stone_played(self, x, y, color):
        self.send_command("play %s %s" % (color_names[color], GTP.str_from_coords(x, y)))

    def generate_move(self, color):
        response = self.send_command("genmove %s" % color_names[color])
        if 'pass' in response.lower():
            return None
        else: 
            return GTP.coords_from_str(response)

    def undo(self):
        self.send_command('undo')

    def quit(self):
        pass

    def final_status_list(self, status):
        return self.send_command("final_status_list %s" % status)


if __name__ == '__main__':
    helper = HelperEngine()

    helper.set_board_size(19)
    helper.clear_board()
    helper.set_komi(6.5)
    helper.stone_played(5, 5, Color.Black)
    move = helper.generate_move(Color.White)
    print "move =", move
    helper.undo()
    move = helper.pick_move(Color.White)
    print "move =", move

