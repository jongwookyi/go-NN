

class TFPolicy:
    def __init__(self, model, threshold_prob):
        self.model = model
        self.threshold_prob = threshold_prob

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def suggest_moves(self, board):
        board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(board, color)
        Normalization.apply_featurewise_normalization_C(board_feature_planes)
        feed_dict = {self.feature_planes: feature_batch}
        move_logits = self.sess.run(self.logits, feed_dict)
        move_probs = softmax(move_logits, softmax_temp)
        # zero out illegal moves
        for x in xrange(self.model.N):
            for y in xrange(self.model.N):
                ind = self.model.N * x + y 
                if not self.board.play_is_legal(x, y, color):
                    logits[ind] = -1e99
        move_probs = softmax(move_logits, softmax_temp)
        sum_probs = np.sum(move_probs)
        if sum_probs == 0: return [] # no legal moves
        move_probs /= sum_probs # re-normalize probabilities

        good_moves = []
        cum_prob = 0.0
        while cum_prob < self.threshold_prob:
            ind = np.argmax(move_probs)
            x,y = ind / self.model.N, ind % self.model.N
            good_moves.append((x,y))
            prob = move_probs[ind]
            cum_prob += prob

        return good_moves


class IdiotPolicy:
    def suggest_moves(self, board):
        moves = []
        for x in xrange(board.N):
            for y in xrange(board.N):
                if board.play_is_legal(x, y, board.color_to_play):
                    moves.append((x,y))
        return moves


class TFEval:
    def __init__(self, model):
        self.model = model

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.score_op = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def evaluate(self, board):
        board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(board, color)
        Normalization.apply_featurewise_normalization_C(board_feature_planes)
        feed_dict = {self.feature_planes: feature_batch}
        score = self.sess.run(self.score_op, feed_dict)
        return score



# want alpha-beta

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




