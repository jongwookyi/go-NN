#!/usr/bin/python

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import os
import random
import time
from datetime import datetime
import Models
from Minibatcher import *


def loss_func(logits, onehot_moves, minibatch_size):
    probs = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(onehot_moves * tf.log(probs))
    cross_entropy_mean = tf.mul(cross_entropy, tf.constant(1.0 / minibatch_size, dtype=tf.float32), name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    # can do weight decay by adding some more losses to the 'losses' collection
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(onehot_moves,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy

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
        

def build_feed_dict(minibatcher, N, feature_planes, onehot_moves):
    loaded_feature_planes, loaded_move_arrs = minibatcher.next_minibatch()
    apply_random_symmetries(loaded_feature_planes, loaded_move_arrs, N)
    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] # NEED TO CHECK ORDER
    assert minibatch_size == loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
    loaded_onehot_moves = np.zeros((minibatch_size, N*N), dtype=np.float32)
    for i in xrange(minibatch_size): loaded_onehot_moves[i, loaded_move_indices[i]] = 1.0
    return { feature_planes: loaded_feature_planes.astype(np.float32),
             onehot_moves: loaded_onehot_moves }


def restore_from_checkpoint(sess, saver, train_dir):
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    print "Trying to restore from checkpoint in dir", checkpoint_dir
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and ckpt.model_checkpoint_path:
        print "Checkpoint file is ", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print "Restored from checkpoint"
    else:
        print "No checkpoint file found"
        assert False
    return step

def optionally_restore_from_checkpoint(sess, saver, train_dir):
    while True:
        response = raw_input("Restore from checkpoint [y/n]? ").lower()
        if response == 'y': 
            return restore_from_checkpoint(sess, saver, train_dir)
        if response == 'n':
            return 0

def add_loss_summaries(total_loss, accuracy):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.99875, name='avg')
    losses = tf.get_collection('losses')
    accuracy_pct = tf.mul(accuracy, tf.constant(100.0), name='accuracy_percent')
    loss_averages_op = loss_averages.apply(losses + [total_loss, accuracy_pct])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss, accuracy_pct]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train_step(total_loss, accuracy, learning_rate):
    loss_averages_op = add_loss_summaries(total_loss, accuracy)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def train_model(model, N, Nfeat, minibatch_size, learning_rate, train_data_dir, val_data_dir, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat], name='feature_planes')
        onehot_moves = tf.placeholder(tf.float32, shape=[None, N*N], name='onehot_moves')

        logits = model.inference(feature_planes, N, Nfeat)
        total_loss, accuracy = loss_func(logits, onehot_moves, minibatch_size)
        train_op = train_step(total_loss, accuracy, learning_rate_ph)

        saver = tf.train.Saver(tf.trainable_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(os.path.join(model.train_dir, 'summaries'), graph_def=sess.graph_def)

        def run_validation(): # run the validation set
            val_minibatcher = NpzMinibatcher(val_data_dir, infinite=False, randomize=False)
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            print "Starting validation..."
            while val_minibatcher.has_more():
                feed_dict = build_feed_dict(val_minibatcher, N, feature_planes, onehot_moves)
                loss_value, accuracy_value = sess.run([total_loss, accuracy], feed_dict=feed_dict)
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)
            summary_writer.add_summary(make_summary('validation_loss', mean_loss), step)
            summary_writer.add_summary(make_summary('validation_accuracy_percent', 100*mean_accuracy), step)
    

        checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
        

        if just_validate: # Just run the validation set once
            restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation()
        else: # Run the training loop
            step = optionally_restore_from_checkpoint(sess, saver, model.train_dir)
            minibatcher = RandomizingNpzMinibatcher(train_data_dir, N, Nfeat, minibatch_size)
            while True:
                if step % 10000 == 0 and step != 0: 
                    run_validation()

                start_time = time.time()
                feed_dict = build_feed_dict(minibatcher, N, feature_planes, onehot_moves)
                load_time = time.time() - start_time
                feed_dict[learning_rate_ph] = learning_rate
    
                start_time = time.time()
                _, loss_value, accuracy_value, summary_str = sess.run([train_op, total_loss, accuracy, summary_op], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                train_time = time.time() - start_time

                if step % 10 == 0:
                    summary_writer.add_summary(summary_str, step)

                if step % 10 == 0:
                    examples_per_sec = minibatch_size / (load_time + train_time)
                    print "%s: step %d, loss = %.2f, accuracy = %.2f%% (%.1f examples/sec), (load=%.3f train=%.3f sec/batch)" % \
                            (datetime.now(), step, loss_value, 100*accuracy_value, examples_per_sec, load_time, train_time)
    
                if step % 1000 == 0 and step != 0:
                    saver.save(sess, os.path.join(model.train_dir, "model.ckpt"))



                step += 1



if __name__ == "__main__":
    #N = 9
    #minibatch_size = 1000
    #Nfeat = 24
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/train" % (minibatch_size, Nfeat)
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/val" % (minibatch_size, Nfeat)
    N = 19
    minibatch_size = 128 #1000
    Nfeat = 24
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb1000_fe24/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb1000_fe24/val"

    print "Training data = %s\nValidation data = %s" % (train_data_dir, val_data_dir)
    
    #model = Models.Linear(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.SingleFull(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.Conv3Full(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.Conv4Full(N, Nfeat, minibatch_size, learning_rate=0.0003)
    #model = Models.Conv5Full(N, Nfeat, minibatch_size, learning_rate=0.00005) # low learning rate for overnight run
    #model = Models.Conv8(N, Nfeat, minibatch_size, learning_rate=0.0003) 
    #model = Models.Conv8(N, Nfeat, minibatch_size, learning_rate=0.0003) 
    #model = Models.Conv8Full(N, Nfeat, minibatch_size, learning_rate=0.00003) 
    #model = Models.Conv12(N, Nfeat, minibatch_size, learning_rate=0.00003) # low learning rate for overnight run (and stability)
    #model = Models.MaddisonMinimal(N, Nfeat, minibatch_size, learning_rate=0.0003) 
    model = Models.Conv6PosDep(N, Nfeat) 

    learning_rate = 0.1
    train_model(model, N, Nfeat, minibatch_size, learning_rate, train_data_dir, val_data_dir, just_validate=False)

