import tensorflow as tf
import numpy as np
import os
from Engine import *
from Board import *
import Features
import Symmetry

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

class InfluenceEngine(BaseEngine):
    def name(self):
        return "InfluenceEngine"

    def version(self):
        return "1.0"

    def __init__(self, model):
        BaseEngine.__init__(self) 
        self.model = model
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
        #        #checkpoint_dir = "/home/greg/coding/ML/go/NN/work/good_checkpoints/conv12posdepELU_N19_fe15"
                restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def make_influence_map(self):
        if self.model.Nfeat == 15:
            board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(self.board, self.board.color_to_play)
            #Normalization.apply_featurewise_normalization_B(board_feature_planes)
        else: 
            assert False
        feature_batch = make_symmetry_batch(board_feature_planes)
        feed_dict = {self.feature_planes: feature_batch}
        logit_batch = self.sess.run(self.logits, feed_dict)
        move_logits = average_logits_over_symmetries(logit_batch, self.model.N)
        move_logits = move_logits.reshape((self.model.N, self.model.N))
        influence_map = np.tanh(move_logits)
        if self.board.color_to_play == Color.White:
            influence_map *= -1
        #influence_map = -1 * np.ones((self.model.N, self.model.N), dtype=np.float32)
        return influence_map


    def pick_move(self, color):
        for i in xrange(10000):
            x = np.random.randint(0, self.board.N-1)
            y = np.random.randint(0, self.board.N-1)
            if self.board.play_is_legal(x, y, color):
                return Move(x,y)
        return Move.Pass()


