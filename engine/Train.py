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
import Features


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

#def influence_loss_func(logits, final_maps):
#    assert False
#    # final maps are originally -1 to 1. rescale them to 0 to 1 probabilities:
#    final_prob_map = final_maps * tf.constant(0.5) + tf.constant(0.5)
#    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=final_prob_map)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='influence_cross_entropy_mean')
#
#    error_rate =  tf.reduce_mean(tf.equal(tf.sign(logits) - tf.sign(final_maps)))
#    accuracy = tf.constant(1.0) - error_rate
#    return cross_entropy_mean, accuracy



def build_feed_dict(minibatcher, minibatch_size, N, feature_planes, onehot_moves):
    loaded_feature_planes, loaded_move_arrs = minibatcher.next_minibatch()

    # approximately normalize inputs. maybe gives a fraction of a percent improvement in training errors, at least early on
    # shift and rescaling calculated on KGS data using make_feature_planes_stones_3liberties_4history_ko
    # haven't tested the difference between this and no normalization when using ELUs
    #loaded_feature_planes = (loaded_feature_planes.astype(np.float32) - 0.154) * 2.77

    # this normalization was worse than the above simple normalization (in my ELU test)
    loaded_feature_planes = \
        (loaded_feature_planes.astype(np.float32) 
         - np.array([0.146, 0.148, 0.706, 0.682, 0.005, 0.018, 0.124, 0.004, 0.018, 0.126, 0.003, 0.003, 0.003, 0.003, 0])) \
         * np.array([2.829, 2.818, 2.195, 2.148, 10, 7.504, 3.0370, 10, 7.576, 3.013, 10, 10, 10, 10, 10])

    loaded_move_arrs = loaded_move_arrs.astype(np.int32) # BIT ME HARD.
    Features.apply_random_symmetries(loaded_feature_planes, loaded_move_arrs)
    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] 
    assert minibatch_size == loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
    loaded_onehot_moves = np.zeros((minibatch_size, N*N), dtype=np.float32)
    for i in xrange(minibatch_size): loaded_onehot_moves[i, loaded_move_indices[i]] = 1.0
    return { feature_planes: loaded_feature_planes.astype(np.float32),
             onehot_moves: loaded_onehot_moves }


#def influence_build_feed_dict(minibatcher, minibatch_size, N, feature_planes, final_maps):
#    assert False
#    loaded_feature_planes, loaded_final_maps = minibatcher.next_minibatch()
#    Features.apply_random_symmetries_influence(loaded_feature_planes, loaded_final_maps)
#    assert minibatch_size == loaded_feature_planes.shape[0] == loaded_move_indices.shape[0]
#    return { feature_planes: loaded_feature_planes.astype(np.float32),
#             onehot_moves: loaded_final_maps.astype(np.float32) }


def restore_from_checkpoint(sess, saver, ema_saver, train_dir):
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    print "Trying to restore from checkpoint in dir", checkpoint_dir
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        print "Checkpoint file is ", checkpoint.model_checkpoint_path
        saver.restore(sess, checkpoint.model_checkpoint_path)
        step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print "Restored from checkpoint"

        ema_dir = os.path.join(train_dir, 'moving_averages')
        ema_checkpoint = tf.train.get_checkpoint_state(ema_dir)
        if ema_checkpoint and ema_checkpoint.model_checkpoint_path:
            print "Moving average checkpoint file is", ema_checkpoint.model_checkpoint_path
            ema_saver.restore(sess, ema_checkpoint.model_checkpoint_path)
            print "Restored moving averages from checkpoint"
        else:
            print "Failed to restore moving averages from checkpoint dir", ema_dir
    else:
        print "No checkpoint file found"
        assert False
    return step

def optionally_restore_from_checkpoint(sess, saver, ema_saver, train_dir):
    while True:
        response = raw_input("Restore from checkpoint [y/n]? ").lower()
        if response == 'y': 
            return restore_from_checkpoint(sess, saver, ema_saver, train_dir)
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

    # Make a Saver for the exponential moving averages
    ema_variables = [loss_averages.average(op) for op in losses + [total_loss, accuracy_pct]]
    ema_saver = tf.train.Saver(ema_variables)

    return loss_averages_op, ema_saver

def train_step(total_loss, accuracy, learning_rate):
    loss_averages_op, ema_saver = add_loss_summaries(total_loss, accuracy)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        #opt = tf.train.AdamOptimizer(learning_rate)
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

    return train_op, ema_saver

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def read_learning_rate(default_lr):
    try: 
        with open('lr.txt', 'r') as f:
            lr = float(f.read().strip())
            return lr
    except:
        print "failed to read learning rate"
        return default_lr

def train_model(model, N, Nfeat, minibatch_size, learning_rate, train_data_dir, val_data_dir, just_validate=False):
    default_learning_rate = learning_rate
    Ngroup = 2
    minibatch_size *= 2
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat], name='feature_planes')
        onehot_moves = tf.placeholder(tf.float32, shape=[None, N*N], name='onehot_moves')

        logits = model.inference(feature_planes, N, Nfeat)
        total_loss, accuracy = loss_func(logits, onehot_moves, minibatch_size)
        train_op, ema_saver = train_step(total_loss, accuracy, learning_rate_ph)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2.0)

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(os.path.join(model.train_dir, 'summaries', datetime.now().strftime('%Y%m%d-%H%M%S')), graph_def=sess.graph_def)

        def run_validation(): # run the validation set
            val_minibatcher = NpzMinibatcher(val_data_dir)
            val_minibatch_size = 128
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            print "Starting validation..."
            while val_minibatcher.has_more():
                if mb_num % 100 == 0: print "validation minibatch #%d (val_minibatch_size = %d)" % (mb_num, val_minibatch_size)
                feed_dict = build_feed_dict(val_minibatcher, val_minibatch_size, N, feature_planes, onehot_moves)
                loss_value, accuracy_value = sess.run([total_loss, accuracy], feed_dict=feed_dict)
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)
            summary_writer.add_summary(make_summary('validation_loss', mean_loss), step)
            summary_writer.add_summary(make_summary('validation_accuracy_percent', 100*mean_accuracy), step)
    

        if just_validate: # Just run the validation set once
            restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation()
        else: # Run the training loop
            step = optionally_restore_from_checkpoint(sess, saver, ema_saver, model.train_dir)
            #minibatcher = RandomizingNpzMinibatcher(train_data_dir)
            minibatcher = GroupingRandomizingNpzMinibatcher(train_data_dir, Ngroup)
            while True:
                if step % 10000 == 0 and step != 0: 
                    run_validation()

                if step % 10 == 0:
                    learning_rate = read_learning_rate(default_learning_rate)
                    summary_writer.add_summary(make_summary('learningrate', learning_rate), step)

                start_time = time.time()
                feed_dict = build_feed_dict(minibatcher, minibatch_size, N, feature_planes, onehot_moves)
                load_time = time.time() - start_time
                feed_dict[learning_rate_ph] = learning_rate
    
                start_time = time.time()
                if step % 10 == 0:
                    _, loss_value, accuracy_value, summary_str = sess.run([train_op, total_loss, accuracy, summary_op], feed_dict=feed_dict)
                else:
                    _, loss_value, accuracy_value              = sess.run([train_op, total_loss, accuracy             ], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                train_time = time.time() - start_time

                if step % 100 == 0 or (step % 10 == 0 and step < 10000):
                    summary_writer.add_summary(summary_str, step)

                if step % 10 == 0:
                    examples_per_sec = minibatch_size / (load_time + train_time)
                    print "%s: step %d, lr=%.6f, loss = %.2f, accuracy = %.2f%% (%.1f examples/sec), (load=%.3f train=%.3f sec/batch)" % \
                            (datetime.now(), step, learning_rate, loss_value, 100*accuracy_value, examples_per_sec, load_time, train_time)
    
                if step % 1000 == 0 and step != 0:
                    saver.save(sess, os.path.join(model.train_dir, "checkpoints", "model.ckpt"), global_step=step)
                    ema_saver.save(sess, os.path.join(model.train_dir, "moving_averages", "moving_averages.ckpt"), global_step=step)


                step += 1



if __name__ == "__main__":
    #N = 9
    #minibatch_size = 1000
    #Nfeat = 24
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/train" % (minibatch_size, Nfeat)
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/CGOS/9x9/processed/mb%d_fe%d/val" % (minibatch_size, Nfeat)
    N = 19
    minibatch_size = 128 #1000
    Nfeat = 15
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb1000_fe24/train-randomized-3"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/mb1000_fe24/val"
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/ps3l4hkNf15-rand2"
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/train-rand-2"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/val-small"
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/first_move_test"
    #val_data_dir = None

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
    #model = Models.Conv6PosDep(N, Nfeat) 
    #model = Models.Conv8PosDep(N, Nfeat) 
    #model = Models.Conv10PosDep(N, Nfeat) 
    model = Models.Conv10PosDepELU(N, Nfeat) 
    #model = Models.Conv12PosDep(N, Nfeat) 
    #model = Models.FirstMoveTest(N, Nfeat) 

    learning_rate = 0.1
    train_model(model, N, Nfeat, minibatch_size, learning_rate, train_data_dir, val_data_dir, just_validate=False)

