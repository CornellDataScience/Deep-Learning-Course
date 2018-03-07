"""
Train a fully connected neural network on the MNIST dataset.
"""

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/')


def weight_variable(shape):
    w_init = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable('w', shape=shape, initializer=w_init)


def bias_variable(shape):
    b_init = tf.constant(0.0, shape=shape)
    return tf.get_variable('b', shape=shape, initializer=b_init)


with tf.variable_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    
    prob_i = tf.placeholder(tf.float32)
    x_drop = tf.nn.dropout(x, keep_prob=prob_i)

with tf.variable_scope('fc1'):
    W_fc1 = weight_variable([784, 1200])
    b_fc1 = bias_variable([1200])
    h_fc1 = tf.nn.relu(tf.matmul(x_drop, W_fc1) + b_fc1)
    
    prob_fc = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, prob_fc)

with tf.variable_scope('fc2'):
    W_fc2 = weight_variable([1200, 1200])
    b_fc2 = bias_variable([1200])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    h_fc2_drop = tf.nn.dropout(h_fc2, prob_fc)

with tf.variable_scope('softmax'):
    W_fc3 = weight_variable([1200, 10])
    b_fc3 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

with tf.variable_scope('optimization'):
    # define the loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y), reduction_indices=[1]))
    
    # define training step and accuracy
    train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(cross_entropy)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# create a saver
saver = tf.train.Saver()
save_path = '../models/mnist_fc'

# initialize the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
batch_size = 100
print('Starting training...')
start_time = time.time()
best_accuracy = 0

for i in range(10000):
    x_train, y_train = mnist.train.next_batch(batch_size)
    if (i + 1) % 1000 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y: y_train, prob_i: 1.0, prob_fc: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

        # validate
        val_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels,
                                                     prob_i: 1.0, prob_fc: 1.0})
        if val_accuracy > best_accuracy:
            saver.save(sess, save_path)
            best_accuracy = val_accuracy
            print("Validation accuracy improved: %g. Saving the network." % val_accuracy)
        else:
            saver.restore(sess, save_path)
            print("Validation accuracy was: %g. Previous accuracy: %g. " % (val_accuracy, best_accuracy) +
                  "Using old parameters for further optimizations.")

    # run training step (note dropout hyperparameters)
    sess.run(train_step, feed_dict={x: x_train, y: y_train, prob_i: 0.8, prob_fc: 0.5})

print("Training took %.4f seconds." % (time.time() - start_time))

# test
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, prob_i: 1.0, prob_fc: 1.0})
print("Best test accuracy: %g" % best_accuracy)