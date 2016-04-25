

# want policy network to influence evaluation????
# could modify score by policy probability, possibly in a depth-dependent way

def minimax_eval(board, policy, value, depth):
    if depth == 0:
        return value.evaluate(board) # TODO

    moves = policy.suggest_moves(board) # TODO
    assert len(moves) > 0
    best_score = -99
    for move in moves:
        board.make_undoable_move(move) # TODO
        score = -1 * minimax_eval(board, policy, value, depth-1)
        board.undo_last_move() # TODO
        if score > best_score: 
            best_score = score
    return best_score

def choose_move_minimax(board, policy, value, depth):
    assert depth > 0

    moves = policy.suggest_moves(board)
    best_score = -99
    best_move = None
    for move in moves:
        board.make_undoable_move(move) # TODO
        score = -1 * minimax_eval(board, policy, value, depth-1)
        board.undo_last_move() # TODO
        if score > best_score: 
            best_score, best_move = score, move
    return best_move


# Return value of position if it's between lower and upper.
# If it's <= lower, return lower; if it's >= upper return upper.
def alphabeta_eval(board, policy, value, lower, upper, depth):
    if depth == 0:
        return value.evaluate(board) # TODO

    moves = policy.suggest_moves(board) # TODO
    assert len(moves) > 0
    for move in moves:
        board.make_undoable_move(move) # TODO
        score = -1 * alphabeta_eval(board, policy, value, -upper, -lower, depth-1)
        board.undo_last_move() # TODO
        if score >= upper: 
            return upper
        if score > lower:
            lower = score
    return lower

def choose_move_alphabeta(board, policy, value, depth):
    assert depth > 0

    moves = policy.suggest_moves(board)
    lower = -1
    upper = +1
    best_move = None
    for move in moves:
        board.make_undoable_move(move) # TODO
        score = -1 * alphabeta_eval(board, policy, value, -upper, -lower, depth-1)
        board.undo_last_move() # TODO
        if score > lower:
            lower, best_move = score, move
    return best_move




