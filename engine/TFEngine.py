import tensorflow as tf
import numpy as np
from Engine import *
from MakeTrainingData import make_feature_planes

class TFEngine(BaseEngine):
    def __init__(self, eng_name, inference, model_ckpoint_dir):
        super(TFEngine,self).__init__() 
        self.Nfeat = 3
        self.N = 19
        self.eng_name = eng_name

        self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.N, self.N, self.Nfeat], name='feature_planes')
        self.logits, variables_to_restore = inference(self.feature_planes)
        saver = tf.train.Saver(variables_to_restore)
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_ckpoint_dir)
        assert ckpt and ckpt.model_checkpoint_path, "No model checkpoint file found"
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def pick_move(self, color):
        board_feature_planes = make_feature_planes(self.board, color)
        board_feature_planes = board_feature_planes.reshape((1, self.N, self.N, self.Nfeat))
        feed_dict = {self.feature_planes: board_feature_planes}

        move_logits = self.sess.run(self.logits, feed_dict)
        move_logits = move_logits.reshape((self.N * self.N,))

        best = None
        best_score = -1e99
        for x in xrange(self.N):
            for y in xrange(self.N):
                ind = self.N * x + y # NEED TO CHECK WHETHER THIS IS CORRECT
                if move_logits[ind] > best_score:
                    if self.board.play_is_legal(x, y, color):
                        best_score = move_logits[ind]
                        best = x,y
        return best


