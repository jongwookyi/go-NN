
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
