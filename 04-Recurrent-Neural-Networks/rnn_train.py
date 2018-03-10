"""
Train a character-level RNN.
"""

import numpy as np
import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, static_rnn


train_file = sys.argv[1]
with open(train_file, 'r') as f:
    transcript = f.read()
transcript = transcript.replace('\n', ' ')
unique_chars = sorted(list(set(transcript)))
num_classes = len(unique_chars)
char2idx = dict(zip(unique_chars, range(num_classes)))

# hyperparameters
num_steps = 50
num_layers = 2
state_size = 128
batch_size = 50
learning_rate = 0.001
max_epochs = 50

# input & embedding layer
x = tf.placeholder(tf.int32, [batch_size, num_steps])
y = tf.placeholder(tf.int32, [batch_size, num_steps])
one_hot_embeddings = tf.eye(num_classes)
x_embed = tf.nn.embedding_lookup(one_hot_embeddings, x)
y_embed = tf.nn.embedding_lookup(one_hot_embeddings, y)
x_inputs = tf.unstack(x_embed, axis=1)
y_outputs = tf.unstack(y_embed, axis=1)

# RNN cells
cells = [BasicLSTMCell(state_size) for _ in range(num_layers)]
cell = MultiRNNCell(cells)
init_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, _ = static_rnn(cell, x_inputs, initial_state=init_state)

# softmax layer
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
preds = [tf.nn.softmax(logit) for logit in logits]

losses = [tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)
          for label, logit in zip(y_outputs, logits)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)
correct = [tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y_output, 1)), tf.float32) for
           pred, y_output in zip(preds, y_outputs)]
accuracy = tf.reduce_mean(correct)

# saving trained model
save_dir = '../models/rnn'
saver = tf.train.Saver()
init = tf.global_variables_initializer()

# pre-process training data
data = ([], [])
print('Pre-processing training data...')
start_time = time.time()
for i in range(len(transcript) - num_steps - 2):
    x_text = transcript[i: i + num_steps]
    y_text = transcript[i + 1: i + num_steps + 1]
    data[0].append([char2idx[char] for char in x_text])
    data[1].append([char2idx[char] for char in y_text])
print('Pre-processed {} characters in {:.2f}s'.format(len(transcript), time.time() - start_time))
data = (np.array(data[0], dtype=int), np.array(data[1], dtype=int))
data = np.hstack(data)
n = data.shape[0]
data_train, data_validate, data_test = np.split(data, [n - 100, n - 50])
x_val, y_val = data_validate[:, :num_steps], data_validate[:, num_steps:]
x_test, y_test = data_test[:, :num_steps], data_test[:, num_steps:]

with tf.Session() as sess:

    print('Starting training...')
    start_time = time.time()
    best_accuracy = 0
    sess.run(init)

    for i in range(max_epochs):
        np.random.shuffle(data_train)
        for j in range(data_train.shape[0] // batch_size):
            start, end = j * batch_size, (j + 1) * batch_size
            x_batch, y_batch = data_train[start:end, :num_steps], data_train[start:end, num_steps:]
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
        val_accuracy = sess.run(accuracy, feed_dict={x: x_val, y: y_val})
        print('Validation accuracy:', val_accuracy)
        if val_accuracy > best_accuracy:
            saver.save(sess, save_dir)
            print('Saving model...')
        else:
            saver.restore(sess, save_dir)
            print('Previous accuracy: {}, restoring model...'.format(best_accuracy))
