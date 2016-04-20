#!/usr/bin/python

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import os
import random
import time
from datetime import datetime
import MoveModels
import MoveTraining
import InfluenceModels
import InfluenceTraining
import NPZ
import Normalization


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
    accuracy_pct = tf.mul(accuracy, tf.constant(100.0), name='accuracy_percent')
    interesting_numbers = [total_loss, accuracy_pct]

    averages = tf.train.ExponentialMovingAverage(0.99875, name='avg')
    averages_op = averages.apply(interesting_numbers)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in interesting_numbers:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, averages.average(l))

    # Make a Saver for the exponential moving averages
    ema_variables = [averages.average(op) for op in interesting_numbers]
    ema_saver = tf.train.Saver(ema_variables)

    return averages_op, ema_saver

def train_step(total_loss, accuracy, learning_rate, momentum=None):
    loss_averages_op, ema_saver = add_loss_summaries(total_loss, accuracy)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        #opt = tf.train.GradientDescentOptimizer(learning_rate)
        #print "USING GRADIENT DESCENT"
        opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        print "USING MOMENTUM"
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

def read_float_from_file(filename, default):
    try: 
        with open(filename, 'r') as f:
            x = float(f.read().strip())
            return x
    except:
        print "failed to read from file", filename, "; using default value", default
        return default

def append_line_to_file(filename, s):
    with open(filename, 'a') as f:
        f.write(s)
        f.write('\n')

def train_model(model, N, Nfeat, build_feed_dict, normalization, loss_func, train_data_dir, val_data_dir, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        momentum_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat])

        logits = model.inference(feature_planes, N, Nfeat)
        outputs, total_loss, accuracy = loss_func(logits)
        print "total_loss =", total_loss
        train_op, ema_saver = train_step(total_loss, accuracy, learning_rate_ph, momentum_ph)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2.0)

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(os.path.join(model.train_dir, 'summaries', datetime.now().strftime('%Y%m%d-%H%M%S')), graph_def=sess.graph_def)

        def run_validation(): # run the validation set
            val_loader = NPZ.Loader(val_data_dir)
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            print "Starting validation..."
            while val_loader.has_more():
                if mb_num % 100 == 0: print "validation minibatch #%d" % mb_num
                feed_dict = build_feed_dict(val_loader, normalization, feature_planes, outputs)
                loss_value, accuracy_value = sess.run([total_loss, accuracy], feed_dict=feed_dict)
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)
            summary_writer.add_summary(make_summary('validation_loss', mean_loss), step)
            summary_writer.add_summary(make_summary('validation_accuracy_percent', 100*mean_accuracy), step)
    
        last_training_loss = None

        if just_validate: # Just run the validation set once
            restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation()
        else: # Run the training loop
            step = optionally_restore_from_checkpoint(sess, saver, ema_saver, model.train_dir)
            #loader = NPZ.RandomizingLoader(train_data_dir)
            loader = NPZ.GroupingRandomizingLoader(train_data_dir, Ngroup=1)
            #loader = NPZ.SplittingRandomizingLoader(train_data_dir, Nsplit=2)
            while True:
                if step % 10000 == 0 and step != 0: 
                    run_validation()

                start_time = time.time()
                feed_dict = build_feed_dict(loader, normalization, feature_planes, outputs)
                load_time = time.time() - start_time

                if step % 10 == 0:
                    learning_rate = read_float_from_file('../work/lr.txt', default=0.1)
                    momentum = read_float_from_file('../work/momentum.txt', default=0.9)
                    summary_writer.add_summary(make_summary('learningrate', learning_rate), step)
                    summary_writer.add_summary(make_summary('momentum', momentum), step)
                feed_dict[learning_rate_ph] = learning_rate
                feed_dict[momentum_ph] = momentum
    
                start_time = time.time()
                if step % 10 == 0:
                    _, loss_value, accuracy_value, summary_str = sess.run([train_op, total_loss, accuracy, summary_op], feed_dict=feed_dict)
                else:
                    _, loss_value, accuracy_value              = sess.run([train_op, total_loss, accuracy            ], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                train_time = time.time() - start_time

                if step % 100 == 0 or (step % 10 == 0 and step < 10000):
                    summary_writer.add_summary(summary_str, step)

                if step % 10 == 0:
                    minibatch_size = feed_dict[feature_planes].shape[0]
                    examples_per_sec = minibatch_size / (load_time + train_time)
                    print "%s: step %d, lr=%.6f, mom=%.2f, loss = %.2f, accuracy = %.2f%% (mb_size=%d, %.1f examples/sec), (load=%.3f train=%.3f sec/batch)" % \
                            (datetime.now(), step, learning_rate, momentum, loss_value, 100*accuracy_value, minibatch_size, examples_per_sec, load_time, train_time)
    
                if step % 1000 == 0 and step != 0:
                    saver.save(sess, os.path.join(model.train_dir, "checkpoints", "model.ckpt"), global_step=step)
                    ema_saver.save(sess, os.path.join(model.train_dir, "moving_averages", "moving_averages.ckpt"), global_step=step)

                step += 1



if __name__ == "__main__":
    N = 19
    #Nfeat = 15
    Nfeat = 21
    
    #model = Models.Conv6PosDep(N, Nfeat) 
    #model = Models.Conv8PosDep(N, Nfeat) 
    #model = Models.Conv10PosDep(N, Nfeat) 
    #model = MoveModels.Conv10PosDepELU(N, Nfeat) 
    #model = MoveModels.Conv12PosDepELU(N, Nfeat) 
    model = MoveModels.Conv12PosDepELUBig(N, Nfeat) 
    #model = MoveModels.Conv16PosDepELU(N, Nfeat) 
    #model = MoveModels.Res5x2PreELU(N, Nfeat) 
    #model = MoveModels.Res10x2PreELU(N, Nfeat) 
    #model = MoveModels.Conv4PosDepELU(N, Nfeat) 
    #model = Models.FirstMoveTest(N, Nfeat) 
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/train-rand-2"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/val-small"
    #normalization = Normalization.apply_featurewise_normalization_B
    train_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/val-small"
    normalization = Normalization.apply_featurewise_normalization_C
    build_feed_dict = MoveTraining.build_feed_dict
    loss_func = MoveTraining.loss_func

    """
    #model = InfluenceModels.Conv4PosDep(N, Nfeat)
    model = InfluenceModels.Conv12PosDepELU(N, Nfeat)
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/val"
    build_feed_dict = InfluenceTraining.build_feed_dict
    loss_func = InfluenceTraining.loss_func
    normalization = Normalization.apply_featurewise_normalization_B
    """

    print "Training data = %s\nValidation data = %s" % (train_data_dir, val_data_dir)

    train_model(model, N, Nfeat, build_feed_dict, normalization, loss_func, train_data_dir, val_data_dir, just_validate=False)

