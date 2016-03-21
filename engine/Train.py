#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import random
import time
from datetime import datetime
from MakeTrainingData import read_minibatch
import Models


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

def train_step(total_loss, learning_rate):
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

def apply_random_symmetries(feature_planes, move_arrs, N):
    for i in range(feature_planes.shape[0]):
        if random.random() < 0.5: # flip x
            feature_planes[i,:,:,:] = feature_planes[i,::-1,:,:]
            move_arrs[i,0] = N - move_arrs[i,0] - 1
        if random.random() < 0.5: # flip y
            feature_planes[i,:,:,:] = feature_planes[i,:,::-1,:]
            move_arrs[i,1] = N - move_arrs[i,1] - 1
        if random.random() < 0.5: # swap x and y
            feature_planes[i,:,:,:] = np.transpose(feature_planes[i,:,:,:], (1,0,2))
            move_arrs[i,:] = move_arrs[i,::-1]
        

def build_feed_dict(mb_filename, N, feature_planes, onehot_moves):
    loaded_feature_planes, loaded_move_arrs = read_minibatch(mb_filename)
    apply_random_symmetries(loaded_feature_planes, loaded_move_arrs, N)
    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] # NEED TO CHECK ORDER
    assert minibatch_size == loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
    loaded_onehot_moves = np.zeros((minibatch_size, N*N), dtype=np.float32)
    for i in xrange(minibatch_size): loaded_onehot_moves[i, loaded_move_indices[i]] = 1.0
    return { feature_planes: loaded_feature_planes.astype(np.float32),
             onehot_moves: loaded_onehot_moves }


def restore_from_checkpoint(sess, saver, ckpt_dir):
    print "Trying to restore from checkpoint in dir", ckpt_dir
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print "Checkpoint file is ", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        print "Restored from checkpoint"
    else:
        print "No checkpoint file found"
        assert False


def train_model(model, N, minibatch_size, Nfeat, train_data_dir, val_data_dir, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat], name='feature_planes')
        onehot_moves = tf.placeholder(tf.float32, shape=[None, N*N], name='onehot_moves')
        logits = model.inference(feature_planes, N, Nfeat)
        loss, accuracy = loss_func(logits, onehot_moves)
        train_op = train_step(loss, learning_rate_ph)

        saver = tf.train.Saver(tf.trainable_variables())

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        def run_validation(): # run the validation set
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            print "Starting validation..."
            for fn in os.listdir(val_data_dir):
                mb_filename = os.path.join(val_data_dir, fn)
                #print "validation minibatch # %d = %s" % (mb_num, mb_filename)
                feed_dict = build_feed_dict(mb_filename, N, feature_planes, onehot_moves)
                loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict=feed_dict)
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)

        if just_validate: # Just run the validation set once
            restore_from_checkpoint(sess, saver, model.checkpoint_dir)
            run_validation()
        else: # Run the training loop
            #restore_from_checkpoint(sess, saver, model.checkpoint_dir)
            queue = RandomizedFilenameQueue(train_data_dir)
            step = 0
            while True:
                if step % 1000 == 0: 
                    run_validation()

                mb_filename = queue.next_filename()
    
                start_time = time.time()
                feed_dict = build_feed_dict(mb_filename, N, feature_planes, onehot_moves)
                load_time = time.time() - start_time
                feed_dict[learning_rate_ph] = model.learning_rate
    
                start_time = time.time()
                _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                train_time = time.time() - start_time
    
                if step % 10 == 0:
                    examples_per_sec = minibatch_size / (load_time + train_time)
                    print "%s: step %d, loss = %.2f, accuracy = %.2f%% (%.1f examples/sec), (load=%.3f train=%.3f sec/batch)" % \
                            (datetime.now(), step, loss_value, 100*accuracy_value, examples_per_sec, load_time, train_time)
    
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(model.checkpoint_dir, "model.ckpt"))
    

                step += 1



if __name__ == "__main__":

    N = 9
    minibatch_size = 1000
    Nfeat = 16
    train_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/train" % (minibatch_size, Nfeat)
    val_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/val" % (minibatch_size, Nfeat)
    #N = 19
    #minibatch_size = 8192
    #Nfeat = 3
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb%d_fe%d/train" % (minibatch_size, Nfeat)
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb%d_fe%d/val" % (minibatch_size, Nfeat)
    #print "Training data = %s\nValidation data = %s" % (train_data_dir, val_data_dir)
    
    #model = Models.Linear(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.SingleFull(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.Conv3Full(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.Conv4Full(N, Nfeat, minibatch_size, learning_rate=0.0003)
    model = Models.Conv5Full(N, Nfeat, minibatch_size, learning_rate=0.00005) # low learning rate for overnight run
    
    train_model(model, N, minibatch_size, Nfeat, train_data_dir, val_data_dir, just_validate=False)

