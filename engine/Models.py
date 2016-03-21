import tensorflow as tf

def conv_layer(inputs, radius, Nin, Nout, stddev=0.1):
    kernel = tf.Variable(tf.truncated_normal([radius, radius, Nin, Nout], stddev=stddev))
    bias = tf.Variable(tf.constant(0.1, shape=[Nout]))
    conv = tf.nn.relu(tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME') + bias)
    return conv

def linear_layer(inputs, Nin, Nout):
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



"""
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
    NK = 32
    Nhidden = 1024
    K_1 = tf.Variable(tf.truncated_normal([5, 5, Nfeat, NK], stddev=0.1), name='K_1')
    b_1 = tf.Variable(tf.constant(0.1, shape=[NK]), name='b_1')
    conv1 = tf.nn.relu(tf.nn.conv2d(feature_planes, K_1, [1, 1, 1, 1], padding='SAME') + b_1)
    K_2 = tf.Variable(tf.truncated_normal([3, 3, NK, NK], stddev=0.1), name='K_2')
    b_2 = tf.Variable(tf.constant(0.1, shape=[NK]), name='b_2')
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, K_2, [1, 1, 1, 1], padding='SAME') + b_2)
    conv2_flat = tf.reshape(conv2, [-1, N*N*NK])
    W_3 = tf.Variable(tf.truncated_normal([N*N*NK, Nhidden], stddev=0.1), name='W_3')
    b_3 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[Nhidden]), name='b_3')
    hidden3 = tf.nn.relu(tf.matmul(conv2_flat, W_3) + b_3)
    W_4 = tf.Variable(tf.truncated_normal([Nhidden, N*N], stddev=0.1), name='W_4')
    b_4 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]), name='b_4')
    logits = tf.matmul(hidden3, W_4) + b_4
    variables_to_restore = [K_1, b_1, K_2, b_2, W_3, b_3, W_4, b_4]
    return logits, variables_to_restore

def inference_conv3_full(feature_planes):
    # recommend 9x9, mbs=1000, adam, lr=0.003
    NK = 32
    Nhidden = 1024
    K_1 = tf.Variable(tf.truncated_normal([5, 5, Nfeat, NK], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv1 = tf.nn.relu(tf.nn.conv2d(feature_planes, K_1, [1, 1, 1, 1], padding='SAME') + b_1)
    K_2 = tf.Variable(tf.truncated_normal([3, 3, NK, NK], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, K_2, [1, 1, 1, 1], padding='SAME') + b_2)
    K_3 = tf.Variable(tf.truncated_normal([3, 3, NK, NK], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.1, shape=[NK]))
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, K_2, [1, 1, 1, 1], padding='SAME') + b_2)
    conv3_flat = tf.reshape(conv3, [-1, N*N*NK])
    W_4 = tf.Variable(tf.truncated_normal([N*N*NK, Nhidden], stddev=0.1))
    b_4 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[Nhidden]))
    hidden4 = tf.nn.relu(tf.matmul(conv3_flat, W_4) + b_4)
    W_5 = tf.Variable(tf.truncated_normal([Nhidden, N*N], stddev=0.1))
    b_5 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[N*N]))
    logits = tf.matmul(hidden4, W_5) + b_5
    variables_to_restore = [K_1, b_1, K_2, b_2, K_3, b_3, W_4, b_4, W_5, b_5]
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
"""
