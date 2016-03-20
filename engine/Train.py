#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import random
import time
from datetime import datetime
from MakeTrainingData import read_minibatch

N = 9 #19
Nfeat = 3
minibatch_size = 1000 #8192
learning_rate = 0.0003
max_steps = 10000000

def build_feed_dict(mb_filename, feature_planes, onehot_moves):
    loaded_feature_planes, loaded_move_arrs = read_minibatch(mb_filename)
    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] # NEED TO CHECK ORDER
    assert minibatch_size == loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
    loaded_onehot_moves = np.zeros((minibatch_size, N*N), dtype=np.float32)
    for i in xrange(minibatch_size): loaded_onehot_moves[i, loaded_move_indices[i]] = 1.0
    return { feature_planes: loaded_feature_planes.astype(np.float32),
             onehot_moves: loaded_onehot_moves }

def inference_linear(feature_planes):
    print "Linear model"
    flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
    weights = tf.Variable(tf.truncated_normal([N*N*Nfeat, N*N], stddev=0.1), name='weights')
    biases = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]), name='biases')
    logits = tf.add(tf.matmul(flat_features, weights, name='weight-multiply'), biases, name='biases-add')
    variables_to_restore = [weights, biases]
    return logits, variables_to_restore

def inference_single_conv(feature_planes):
    print "Single convolution"
    kernel = tf.Variable(tf.truncated_normal([5, 5, Nfeat, 1], stddev=0.1), name='kernel')
    conv = tf.nn.conv2d(feature_planes, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
    conv_flat = tf.reshape(conv, [-1, N*N], name='conv_flat')
    biases = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]), name='biases')
    logits = tf.add(conv_flat, biases, name='biases-add')
    variables_to_restore = [kernel, biases]
    return logits, variables_to_restore

def inference_two_convs(feature_planes):
    print "Two convolutions"
    NK = 16
    K_1 = tf.Variable(tf.truncated_normal([5, 5, Nfeat, NK], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv1 = tf.nn.relu(tf.nn.conv2d(feature_planes, K_1, [1, 1, 1, 1], padding='SAME') + b_1)
    K_2 = tf.Variable(tf.truncated_normal([3, 3, NK, 1], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.1, shape=[1]))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, K_2, [1, 1, 1, 1], padding='SAME') + b_2)
    conv2_flat = tf.reshape(conv2, [-1, N*N])
    biases = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]))
    logits = conv2_flat + biases
    variables_to_restore = [K_1, K_2, biases]
    return logits, variables_to_restore

def inference_conv_conv_full(feature_planes):
    # recommend 9x9, mbs=1000, adam, lr=0.003
    NK = 16
    Nhidden = 1024
    K_1 = tf.Variable(tf.truncated_normal([5, 5, Nfeat, NK], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv1 = tf.nn.relu(tf.nn.conv2d(feature_planes, K_1, [1, 1, 1, 1], padding='SAME') + b_1)
    K_2 = tf.Variable(tf.truncated_normal([3, 3, NK, NK], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, K_2, [1, 1, 1, 1], padding='SAME') + b_2)
    conv2_flat = tf.reshape(conv2, [-1, N*N*NK])
    W_3 = tf.Variable(tf.truncated_normal([N*N*NK, Nhidden], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[Nhidden]))
    hidden3 = tf.nn.relu(tf.matmul(conv2_flat, W_3) + b_3)
    W_4 = tf.Variable(tf.truncated_normal([Nhidden, N*N], stddev=0.1))
    b_4 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]))
    logits = tf.matmul(hidden3, W_4) + b_4
    variables_to_restore = [K_1, b_1, K_2, b_2, W_3, b_3, W_4, b_4]
    return logits, variables_to_restore

def inference_single_full(feature_planes):
    # recommend 9x9, mbs=1000, adam, lr=0.003
    Nhidden = 1024
    flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
    W_1 = tf.Variable(tf.truncated_normal([N*N*Nfeat, Nhidden], stddev=0.01), name='W_1')
    b_1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[Nhidden]), name='b_1')
    hidden = tf.nn.relu(tf.matmul(flat_features, W_1) + b_1)
    W_2 = tf.Variable(tf.truncated_normal([Nhidden, N*N], stddev=0.01), name='W_2')
    b_2 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]), name='b_2')
    logits = tf.matmul(hidden, W_2) + b_2
    variables_to_restore = [W_1, b_1, W_2, b_2]
    return logits, variables_to_restore

def inference_full_full(feature_planes):
    Nhidden1 = 512
    Nhidden2 = 512
    flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
    W_1 = tf.Variable(tf.truncated_normal([N*N*Nfeat, Nhidden1], stddev=0.1), name='W_1')
    b_1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[Nhidden1]), name='b_1')
    hidden1 = tf.nn.relu(tf.matmul(flat_features, W_1) + b_1)
    W_2 = tf.Variable(tf.truncated_normal([Nhidden1, Nhidden2], stddev=0.1), name='W_2')
    b_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[Nhidden2]), name='b_2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W_2) + b_2)
    W_3 = tf.Variable(tf.truncated_normal([Nhidden2, N*N], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[N*N]))
    logits = tf.matmul(hidden2, W_3) + b_3
    variables_to_restore = [W_1, b_1, W_2, b_2, W_3, b_3]
    return logits, variables_to_restore

def loss_func(logits, onehot_moves):
    #labels = tf.cast(move_index, tf.int64)
    probs = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(onehot_moves * tf.log(probs))
    cross_entropy_mean = cross_entropy * tf.constant(1.0 / minibatch_size, dtype=tf.float32)
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # can do weight decay by adding some more losses to the 'losses' collection
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(onehot_moves,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy

def train_step(total_loss, global_step):
    # could use global_step for a decaying learning rate
    #return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    return tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

class RandomizedFilenameQueue:
    def __init__(self, base_dir):
        self.queue = []
        self.base_dir = base_dir

    def next_filename(self):
        if not self.queue:
            self.queue = ['%s/%s' % (self.base_dir, f) for f in os.listdir(self.base_dir)]
            random.shuffle(self.queue)
            #print "FilenameQueue: prepared randomly ordered queue of %d files from %s" % (len(self.queue), self.base_dir)
        return self.queue.pop()

def restore_from_checkpoint(sess, saver, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print "Restored from checkpoint %s" % global_step
    else:
        print "No checkpoint file found"
        assert False


def train_model(inference, train_data_dir, val_data_dir, train_ckpt_dir):
    queue = RandomizedFilenameQueue(train_data_dir)

    with tf.Graph().as_default():
        learning_rate_ph = tf.placeholder(tf.float32)
    
        global_step = tf.Variable(0, trainable=False)

        # build the graph
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat], name='feature_planes')
        onehot_moves = tf.placeholder(tf.float32, shape=[None, N*N], name='onehot_moves')
        logits, variables_to_restore = inference(feature_planes)
        loss, accuracy = loss_func(logits, onehot_moves)
        train_op = train_step(loss, global_step)

        saver = tf.train.Saver(variables_to_restore)

        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        restore_from_checkpoint(sess, saver, train_ckpt_dir)

        for step in xrange(max_steps):
            mb_filename = queue.next_filename()

            start_time = time.time()
            feed_dict = build_feed_dict(mb_filename, feature_planes, onehot_moves)
            feed_dict[learning_rate_ph] = learning_rate
            load_time = time.time() - start_time

            start_time = time.time()
            _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
            train_time = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                examples_per_sec = minibatch_size / (load_time + train_time)
                print "%s: step %d, loss = %.2f, accuracy = %.2f%% (%.1f examples/sec), (load=%.3f train=%.3f sec/batch)" % \
                        (datetime.now(), step, loss_value, 100*accuracy_value, examples_per_sec, load_time, train_time)

            if step % 10000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_ckpt_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0: 
                mean_loss = 0.0
                mean_accuracy = 0.0
                mb_num = 0
                for fn in os.listdir(val_data_dir):
                    mb_filename = os.path.join(val_data_dir, fn)
                    #print "validation minibatch # %d = %s" % (mb_num, mb_filename)
                    feed_dict = build_feed_dict(mb_filename, feature_planes, onehot_moves)
                    loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict=feed_dict)
                    mean_loss += loss_value
                    mean_accuracy += accuracy_value
                    mb_num += 1
                mean_loss /= mb_num
                mean_accuracy /= mb_num
                print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)



def val_model(inference, val_data_dir, train_ckpt_dir):
    with tf.Graph().as_default():
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat], name='feature_planes')
        onehot_moves = tf.placeholder(tf.float32, shape=[None, N*N], name='onehot_moves')
        logits, variables_to_restore = inference(feature_planes)
        loss, accuracy = loss_func(logits, onehot_moves)

        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            restore_from_checkpoint(sess, saver, train_ckpt_dir)

            total_load_time = 0.0
            total_eval_time = 0.0
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            for fn in os.listdir(val_data_dir):
                mb_filename = os.path.join(val_data_dir, fn)
                print "validation minibatch # %d = %s" % (mb_num, mb_filename)
                
                start_time = time.time()
                feed_dict = build_feed_dict(mb_filename, feature_planes, onehot_moves)
                total_load_time += time.time() - start_time

                start_time = time.time()
                loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict=feed_dict)
                total_eval_time += time.time() - start_time
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)
            print "total load time = %.3f seconds" % total_load_time
            print "total eval time = %.3f seconds" % total_eval_time


if __name__ == "__main__":
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb8192_fe3/train"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb8192_fe3/val"
    train_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb1000_fe3/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb1000_fe3/val"
    
    #train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_linear"
    #train_model(inference_linear, train_data_dir, val_data_dir, train_ckpt_dir)
    #val_model(inference_linear, val_data_dir, train_ckpt_dir)
    
    #train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_single_conv"
    #train_model(inference_single_conv, train_data_dir, train_ckpt_dir)
    #val_model(inference_single_conv, val_data_dir, train_ckpt_dir)
    
    #train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_single_full"
    #train_model(inference_single_full, train_data_dir, val_data_dir, train_ckpt_dir)
    #val_model(inference_single_full, val_data_dir, train_ckpt_dir)

    #train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_full_full"
    #train_model(inference_full_full, train_data_dir, val_data_dir, train_ckpt_dir)
    #val_model(inference_full_full, val_data_dir, train_ckpt_dir)
    
    #train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_two_convs"
    #train_model(inference_two_convs, train_data_dir, val_data_dir, train_ckpt_dir)
    #val_model(inference_two_convs, val_data_dir, train_ckpt_dir)

    train_ckpt_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv_conv_full"
    train_model(inference_conv_conv_full, train_data_dir, val_data_dir, train_ckpt_dir)
    #val_model(inference_conv_conv_full, val_data_dir, train_ckpt_dir)

