"""
Model architecture for CNN.
"""

import tensorflow as tf


def variable(name, shape, stddev, decay):
    """Creates a variable with weight decay (l2 regularization)."""
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_decay')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_layer(inputs, k_size, n_filters, stddev, decay, scope):
    """Creates a convolutional layer."""
    with tf.variable_scope(scope):
        depth = inputs.get_shape().as_list()[-1]
        kernel = variable('w', [k_size, k_size, depth, n_filters], stddev, decay)
        biases = tf.get_variable('b', shape=[n_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(conv, biases))


def fc_layer(inputs, n_units, stddev, decay, scope):
    """Creates a fully connected layer"""
    with tf.variable_scope(scope):
        units_in = inputs.get_shape().as_list()[-1]
        weights = variable('w', [units_in, n_units], stddev, decay)
        biases = tf.get_variable('b', shape=[n_units], initializer=tf.constant_initializer(0.1))
        return tf.nn.relu(tf.nn.xw_plus_b(inputs, weights, biases))


class CNN:
    def __init__(self, images):

        conv1 = conv_layer(images, 5, 64, 0.005, None, 'conv1')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='norm1')

        conv2 = conv_layer(norm1, 5, 64, 0.005, None, 'conv2')
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        reshape = tf.reshape(pool2, [-1, 2304])
        fc3 = fc_layer(reshape, 384, 0.04, 0.004, 'fc3')
        fc4 = fc_layer(fc3, 192, 0.04, 0.004, 'fc4')

        with tf.variable_scope('softmax'):
            w = variable('w', [192, 10], 1/192, None)
            b = tf.get_variable('b', shape=[10], initializer=tf.constant_initializer(0.0))

        self.logits = tf.nn.xw_plus_b(fc4, w, b)
        self.preds = tf.nn.softmax(self.logits)
        self.labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        correct = tf.equal(tf.argmax(self.preds, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
