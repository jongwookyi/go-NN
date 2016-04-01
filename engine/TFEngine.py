import tensorflow as tf
import numpy as np
import random
import os
from Engine import *
import Book
import Features
import Symmetry

#def build_feed_dict(mb_filename, feature_planes, onehot_moves):
#    N = 9
#    loaded_feature_planes, loaded_move_arrs = read_minibatch(mb_filename)
#    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] # NEED TO CHECK ORDER
#    assert loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
#    minibatch_size = loaded_feature_planes.shape[0]
#    loaded_onehot_moves = np.zeros((minibatch_size, N*N), dtype=np.float32)
#    for i in xrange(minibatch_size): loaded_onehot_moves[i, loaded_move_indices[i]] = 1.0
#    return { feature_planes: loaded_feature_planes.astype(np.float32),
#             onehot_moves: loaded_onehot_moves }

def restore_from_checkpoint(sess, saver, ckpt_dir):
    print "Trying to restore from checkpoint in dir", ckpt_dir
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print "Checkpoint file is ", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print "Restored from checkpoint %s" % global_step
    else:
        print "No checkpoint file found"
        assert False

def softmax(E, temp):
    #print "E =\n", E
    expE = np.exp(temp * (E - max(E))) # subtract max to avoid overflow
    return expE / np.sum(expE)

def sample_from(probs):
    cumsum = np.cumsum(probs)
    r = random.random()
    for i in xrange(len(probs)):
        if r <= cumsum[i]: 
            return i
    assert False, "problem with sample_from" 

def make_symmetry_batch(features):
    assert len(features.shape) == 3
    #print "features[:,:,1] =\n", features[:,:,1]
    N = features.shape[0]
    Nfeat = features.shape[2]
    feature_batch = np.empty((8, N, N, Nfeat), dtype=features.dtype)
    for s in xrange(8):
        feature_batch[s,:,:,:] = features
        Symmetry.apply_symmetry_planes(feature_batch[s,:,:,:], s)
        #print "feature_batch[%d,:,:,1] =\n" % s, feature_batch[s,:,:,1]
    return feature_batch

def print_plane(plane):
    assert False
    for y in xrange(plane.shape[0]):
        for x in xrange(plane.shape[1]):
            print "%7.3f" % plane[x,y],
        print

def print_planes(planes):
    for p in xrange(planes.shape[0]):
        print "PLANE", p
        print_plane(planes[p,:,:])

def average_logits_over_symmetries(logits, N):
    #print "logits.shape =", logits.shape
    assert logits.shape == (8, N*N)
    logit_planes = logits.reshape((8, N, N))
    #print "before inverting symmetries, logit_planes ="
    #print_planes(logit_planes)
    for s in xrange(8):
        Symmetry.invert_symmetry_plane(logit_planes[s,:,:], s)
    #print "after inverting symmetries, logit planes ="
    #print_planes(logit_planes)
    mean_logits = logit_planes.mean(axis=0)
    #print "mean_logits =\n"
    #print_plane(mean_logits)
    mean_logits = mean_logits.reshape((N*N,))
    return mean_logits

def get_book_move(board, book):
    pos_record = Book.lookup_position(book, board)
    if pos_record:
        print "known moves:"
        best_vertex = None
        total_count = 0
        for vertex in pos_record.moves:
            move_record = pos_record.moves[vertex]
            print vertex, " - wins=", move_record.wins, "; losses=", move_record.losses
            total_count += move_record.wins + move_record.losses
        if True: #total_count >= 10:
            min_count = total_count / 10
            popular_moves = [move for move in pos_record.moves if (pos_record.moves[move].wins + pos_record.moves[move].losses > min_count)]
            print "popular moves are", popular_moves
            return random.choice(popular_moves)
    return None



class TFEngine(BaseEngine):
    def __init__(self, eng_name, model):
        super(TFEngine,self).__init__() 
        self.eng_name = eng_name
        self.model = model
        self.book = Book.load_GoGoD_book()

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                #checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                checkpoint_dir = "/home/greg/coding/ML/go/NN/work/good_checkpoints/conv10posdepELU_N19_fe15"
                restore_from_checkpoint(self.sess, saver, checkpoint_dir)


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def pick_move(self, color):
        if self.opponent_passed: return None # Pass if opponent passes????

        book_move = get_book_move(self.board, self.book)
        if book_move:
            print "playing book move", book_move
            return book_move
        print "no book move"

        #board_feature_planes = make_feature_planes(self.board, color)
        board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(self.board, color)
        #board_feature_planes = board_feature_planes.reshape((1, self.model.N, self.model.N, self.model.Nfeat))
        #feed_dict = {self.feature_planes: board_feature_planes}
        feature_batch = make_symmetry_batch(board_feature_planes)
        #print "feature_batch.shape =", feature_batch.shape

        feature_batch = \
            (feature_batch.astype(np.float32) 
             - np.array([0.146, 0.148, 0.706, 0.682, 0.005, 0.018, 0.124, 0.004, 0.018, 0.126, 0.003, 0.003, 0.003, 0.003, 0])) \
            * np.array([2.829, 2.818, 2.195, 2.148, 10, 7.504, 3.0370, 10, 7.576, 3.013, 10, 10, 10, 10, 10])


        feed_dict = {self.feature_planes: feature_batch}

        #move_logits = self.sess.run(self.logits, feed_dict)
        #move_logits = move_logits.reshape((self.model.N * self.model.N,))
        logit_batch = self.sess.run(self.logits, feed_dict)
        move_logits = average_logits_over_symmetries(logit_batch, self.model.N)
        #print move_logits
        softmax_temp = 10.0
        move_probs = softmax(move_logits, softmax_temp)

        #for y in xrange(self.model.N):
        #    for x in xrange(self.model.N):
        #        ind = self.model.N * x + y 
        #        print "%7.3f" % move_logits[ind],
        #    print

        # zero out illegal moves
        for x in xrange(self.model.N):
            for y in xrange(self.model.N):
                ind = self.model.N * x + y 
                if not self.board.play_is_legal(x, y, color):
                    move_probs[ind] = 0
        sum_probs = np.sum(move_probs)
        if sum_probs == 0: return None # no legal moves, pass
        move_probs /= sum_probs # re-normalize probabilities

        pick_best = True
        if pick_best:
            move_ind = np.argmax(move_probs)
        else:
            move_ind = sample_from(move_probs)
        move_x = move_ind / self.model.N
        move_y = move_ind % self.model.N
        return move_x, move_y





