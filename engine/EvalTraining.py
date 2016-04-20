import tensorflow as tf
import numpy as np
import random
import Symmetry

def apply_random_symmetries(many_feature_planes):
    N = many_feature_planes.shape[1]
    for i in range(many_feature_planes.shape[0]):
        s = random.randint(0, 7)
        Symmetry.apply_symmetry_planes(many_feature_planes[i,:,:,:], s)


def build_feed_dict(loader, apply_normalization, feature_planes, final_scores):
    batch = loader.next_minibatch(('feature_planes', 'final_scores'))
    loaded_feature_planes = batch['feature_planes'].astype(np.float32)
    loaded_scores = batch['final_scores'].astype(np.int32) # BIT ME HARD.

    apply_normalization(loaded_feature_planes)

    apply_random_symmetries(loaded_feature_planes)

    N = loaded_feature_planes.shape[1]

    return { feature_planes: loaded_feature_planes,
             final_scores: loaded_scores }

def loss_func(score_op):
    final_scores = tf.placeholder(tf.float32, shape=[None])

    mean_sq_err = tf.reduce_mean(tf.square(score_op - final_scores), 'mean_sq_err')

    correct_prediction = tf.equal(tf.sign(score_op), tf.sign(final_scores))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return final_scores, mean_sq_err, accuracy



