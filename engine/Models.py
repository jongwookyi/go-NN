import tensorflow as tf
import math

def conv_layer_no_relu(inputs, diameter, Nin, Nout):
    fan_in = diameter * diameter * Nin
    stddev = math.sqrt(2.0 / fan_in)
    kernel = tf.Variable(tf.truncated_normal([diameter, diameter, Nin, Nout], stddev=stddev))
    bias = tf.Variable(tf.constant(0.1, shape=[Nout]))
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME') + bias
    return conv

def conv_layer(inputs, diameter, Nin, Nout):
    return tf.nn.relu(conv_layer_no_relu(inputs, diameter, Nin, Nout))

def linear_layer(inputs, Nin, Nout):
    stddev = math.sqrt(2.0 / Nin)
    print "linear layer using stddev =", stddev
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[Nout]))
    out = tf.matmul(inputs, weights) + bias
    return out

def fully_connected_layer(inputs, Nin, Nout):
    return tf.nn.relu(linear_layer(inputs, Nin, Nout))


class Linear:
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_linear_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
        logits = linear_layer(flat_features, N*N*Nfeat, N*N)
        return logits



class SingleFull: # recommend 9x9, mbs=1000, adam, lr=0.003
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_single_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        Nhidden = 1024
        flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
        hidden = fully_connected_layer(flat_features, N*N*Nfeat, Nhidden)
        logits = linear_layer(hidden, Nhidden, N*N)
        return logits

class Conv3Full: # recommend 9x9, mbs=1000, adam, lr=0.003
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv3_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 32
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv3_flat = tf.reshape(conv3, [-1, N*N*NK])
        hidden4 = fully_connected_layer(conv3_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden4, Nhidden, N*N)
        return logits

class Conv4Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv4_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 64
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv4_flat = tf.reshape(conv4, [-1, N*N*NK])
        hidden5 = fully_connected_layer(conv4_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden5, Nhidden, N*N)
        return logits

class Conv5Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv5_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv5 = conv_layer(conv4, 3, NK, NK, stddev=0.01)
        conv5_flat = tf.reshape(conv5, [-1, N*N*NK])
        hidden6 = fully_connected_layer(conv5_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden6, Nhidden, N*N)
        return logits

class Conv8: 
    def __init__(self, N, Nfeat, minibatch_size=1000, learning_rate=0.0003):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv8_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv5 = conv_layer(conv4, 3, NK, NK, stddev=0.01)
        conv6 = conv_layer(conv5, 3, NK, NK, stddev=0.01)
        conv7 = conv_layer(conv6, 3, NK, NK, stddev=0.01)
        conv8 = conv_layer(conv7, 1, NK, 1, stddev=0.01) # todo: switch to no_relu
        conv8_flat = tf.reshape(conv8, [-1, N*N])        
        bias = tf.Variable(tf.constant(0, shape=[N*N], dtype=tf.float32)) # position-dependent bias
        logits = conv8_flat + bias
        return logits

class Conv8Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv8_full_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv4 = conv_layer(conv3, 3, NK, NK)
        conv5 = conv_layer(conv4, 3, NK, NK)
        conv6 = conv_layer(conv5, 3, NK, NK)
        conv7 = conv_layer(conv6, 3, NK, NK)
        conv8 = conv_layer(conv7, 3, NK, NK)
        conv8_flat = tf.reshape(conv8, [-1, N*N*NK])        
        hidden9 = fully_connected_layer(conv8_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden9, Nhidden, N*N)
        return logits

class Conv12: # AlphaGo architecture
    def __init__(self, N, Nfeat, minibatch_size=1000, learning_rate=0.0003):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv12_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv4 = conv_layer(conv3, 3, NK, NK)
        conv5 = conv_layer(conv4, 3, NK, NK)
        conv6 = conv_layer(conv5, 3, NK, NK)
        conv7 = conv_layer(conv6, 3, NK, NK)
        conv8 = conv_layer(conv7, 3, NK, NK)
        conv9 = conv_layer(conv8, 3, NK, NK)
        conv10 = conv_layer(conv9, 3, NK, NK)
        conv11 = conv_layer(conv10, 3, NK, NK)
        conv12 = conv_layer_no_relu(conv11, 1, NK, 1)
        conv12_flat = tf.reshape(conv12, [-1, N*N])        
        bias = tf.Variable(tf.constant(0, shape=[N*N], dtype=tf.float32)) # position-dependent bias
        logits = conv12_flat + bias
        return logits


