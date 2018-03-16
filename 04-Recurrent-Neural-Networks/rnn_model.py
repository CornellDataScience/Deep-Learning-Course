"""
TensorFlow implementation of a character-level RNN.
Adapted from https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq


class RNN:
    def __init__(self, args, training=True):
        """Initialize RNN model"""
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        cell_fn = rnn.GRUcell
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnn'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])

        with tf.name_scope('loss'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)

        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prompt='The '):
        """Generate predictions from RNN.

        Args:
            sess (tf.Session): A tensorflow session containing the RNN graph from which to generate predictions
            chars (dict):      A dictionary mapping indices to characters (e.g. {1: 'a', 2: 'b'})
            vocab (dict):      A dictionary mapping characters to indices; inverse of chars (e.g. {'a': 1, 'b': 2})
            num (int):         Length of string to generate (not including prompt)
            prompt (str):      Prompt to start generation (i.e. beginning of predicted text)

        Returns:
            str:               RNN-generated text
        """
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prompt[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        output = prompt
        char = prompt[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            [probs, state] = sess.run([self.probs, self.final_state], {self.input_data: x, self.initial_state: state})
            p = probs[0]
            sample = weighted_pick(p)
            pred = chars[sample]
            output += pred
            char = pred
        return output
