import tensorflow as tf
import numpy as np
import random
import os
from Engine import *
import Features

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
    expE = np.exp(temp * (E - max(E))) # subtract max to avoid overflow
    return expE / np.sum(expE)

def sample_from(probs):
    cumsum = np.cumsum(probs)
    r = random.random()
    for i in xrange(len(probs)):
        if r <= cumsum[i]: 
            return i
    assert False, "problem with sample_from" 


class TFEngine(BaseEngine):
    def __init__(self, eng_name, model):
        super(TFEngine,self).__init__() 
        self.eng_name = eng_name
        self.model = model

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                restore_from_checkpoint(self.sess, saver, os.path.join(model.train_dir, 'checkpoints_original'))


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def pick_move(self, color):
        if self.opponent_passed: return None # Pass if opponent passes????

        #board_feature_planes = make_feature_planes(self.board, color)
        board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(self.board, color)
        board_feature_planes = board_feature_planes.reshape((1, self.model.N, self.model.N, self.model.Nfeat))
        feed_dict = {self.feature_planes: board_feature_planes}

        move_logits = self.sess.run(self.logits, feed_dict)
        move_logits = move_logits.reshape((self.model.N * self.model.N,))
        #print move_logits
        softmax_temp = 10.0
        move_probs = softmax(move_logits, softmax_temp)

        for y in xrange(self.model.N):
            for x in xrange(self.model.N):
                ind = self.model.N * x + y 
                print "%7.3f" % move_logits[ind],
            print

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





