# Attempt at vanilla-rnn from scratch on tensorflow
# Shubhom Bhattacharya
import tensorflow as tf
import pandas as pd
import nltk
from nltk import word_tokenize
import numpy as np

seed = 31337
np.random.seed(seed)


def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))
    cnt=0
    for s in data_:
        v = [0.0]*len(vocab)
        v[int(np.where(vocab == s)[0])] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def embed_output(data, vocab):
    result = np.empty(len(data))
    for idx, word in enumerate(data):
        result[idx] = int(np.where(vocab == data[idx])[0])  # Get index of word in vocab array
    return result


# Load alice in wonderland into an array of numbers called `alice_embed`
alice_load = np.load("alice_arrays.npz")
alice_embed = embed_output(alice_load['words'], alice_load['vocab'])  # Alice as sequence of ints, shape [num_words]
embedding_size = len(alice_load['vocab'])


# Split
sequence_length = 50
truncated_length = (len(alice_embed)//sequence_length) * sequence_length  # Need to make everything divisible
alice_embed = alice_embed[:truncated_length]
alice_split = np.reshape(alice_embed, (-1, sequence_length))  # break the text into fixed length sequences
print(alice_split.shape)
num_sequences = alice_split.shape[0]
indices = np.random.permutation(num_sequences)
pct_train = 0.9
training_idx, test_idx = indices[:int(pct_train*num_sequences)], indices[int(pct_train*num_sequences):]
alice_train, alice_test = alice_split[training_idx, :], alice_split[test_idx, :]

num_units = 256  # number of RNN units in a RNNCell
num_layers = 3

with tf.Graph().as_default():
    tf.set_random_seed(seed)

    x = tf.placeholder(shape=[None, None], dtype=tf.int32)  # Input sequence placeholder [BATCHSIZE, SEQLENGTH]
    x_hot = tf.one_hot(x, depth=embedding_size)  # one-hot encode x
    batch_size_tensor = tf.shape(x)[0]  # This is a scalar tensor

    num_steps = sequence_length

    with tf.variable_scope('Unrolled') as scope:
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

        # The state of the RNN is the "zero state" at the start of every sequence. This is the initial state
        state = rnn_cell.zero_state(batch_size=batch_size_tensor, dtype=tf.float32)

        # Unroll the graph num_steps back into the "past"
        outputs = []  # python list of tensors so we can keep track of each timestep
        for i in range(num_steps):  # need to unroll up to num_steps units back in time
            if i > 0: scope.reuse_variables()  # Reuse the parameters created in the 1st RNN cell
            output, state = rnn_cell(x_hot[:, i, :], state)  # Step the RNN through the sequence
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1, name='Outputs')

    with tf.name_scope('Softmax'):
        w = tf.get_variable(
            name='Weight',
            initializer=tf.truncated_normal([rnn_cell.output_size, embedding_size], stddev=0.01))
        b = tf.get_variable(name='Bias', initializer=tf.zeros(embedding_size))

        flattened = tf.reshape(outputs, (-1, rnn_cell.output_size))  # Broadcasting doesn't work properly for tf.matmul
        matmul = tf.reshape(tf.matmul(flattened, w), shape=(-1, num_steps, embedding_size))
        scores = tf.add(matmul, b, name='Scores')
        softmax = tf.nn.softmax(scores, name='Softmax')

    # with tf.name_scope('x-state'):
    #     W = tf.get_variable(name='weight',initializer=tf.random_normal(shape=[1,1]),regularizer=None)
    #     b1 = tf.get_variable (name='bias',initializer = tf.constant_initializer()) #necessary?
    # with tf.name_scope ('state'):
    #     U = tf.get_variable(name='weight',initializer=tf.random_normal(shape=[1,1]),regularizer=None)
    #     b2 = tf.get_variable (name='bias',initializer = tf.constant_initializer()) #necessary?
    # with tf.name_scope ('x'):
    #     V = tf.get_variable(name='weight', initializer=tf.random_normal(shape=[1, 1]), regularizer=None)
    #     b3 = tf.get_variable(name='bias', initializer=tf.constant_initializer())  #necessary?
    #
    # ht = tf.nn.tanh(b1 + tf.matmul(x,W)+tf.matmul(ht,U)) #current state is a function of x and previous state
    # out = tf.nn.softmax (b3 + tf.matmul(ht,V)) #output is a function of current state

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=x))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session().as_default() as sess:
        num_epochs = 10000  # number of training epochs
        batch_size = 100

        training_size = alice_train.shape[0]

        sess.run(tf.global_variables_initializer())

        count = 0
        for epoch in range(num_epochs):
            perm = np.random.permutation(training_size)  # Every epoch, get a new set of batches
            for i in range(0, training_size, batch_size):
                idx = perm[i:i + batch_size]  # Select indices for batch
                x_batch = alice_train[idx]
                _, batch_loss = sess.run([train_step, loss], feed_dict={x: x_batch})
                print("epoch %6d, loss=%6f" % (epoch + 1, batch_loss))


