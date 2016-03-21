import tensorflow as tf
import numpy as np
from Engine import *
from MakeTrainingData import make_feature_planes

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


class TFEngine(BaseEngine):
    def __init__(self, eng_name, model):
        super(TFEngine,self).__init__() 
        self.Nfeat = 16 # fixme: these should be stored in model
        self.N = 9
        self.eng_name = eng_name

        # build the graph
        with tf.Graph().as_default():
            self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.N, self.N, self.Nfeat], name='feature_planes')
            self.logits = model.inference(self.feature_planes, self.N, self.Nfeat)
            saver = tf.train.Saver(tf.trainable_variables())
            init = tf.initialize_all_variables()
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            self.sess.run(init)
            restore_from_checkpoint(self.sess, saver, model.checkpoint_dir)


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def pick_move(self, color):
        if self.opponent_passed: return None # Pass if opponent passes????

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


