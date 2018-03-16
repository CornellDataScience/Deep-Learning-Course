"""
Train the character-level RNN described in rnn_model.py
"""

import argparse
import collections
import pickle
import numpy as np
import tensorflow as tf
import time
import os
from rnn_model import RNN


def train(args):
    data = Loader(args.data_path, args.batch_size, args.seq_length)
    args.vocab_size = data.vocab_size

    # resume training from existing model (if it exists)
    if args.load_from is not None:
        ckpt = tf.train.get_checkpoint_state(args.load_from)

    # create save directory if it does not already exist
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    # save configurations of current model
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    # save all characters and mapping from characters to index
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data.chars, data.vocab), f)

    # instantiate and train RNN model
    print('Instantiating model...')
    model = RNN(args)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.load_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        print('Starting training...')
        for ep in range(args.num_epochs):
            # decreasing learning rate
            sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.decay_rate ** ep)))
            state = sess.run(model.initial_state)

            for i in range(data.n_batches):
                start = time.time()
                x, y = data.next_batch()
                feed = {model.input_data: x, model.targets: y}

                # assign initial state from previous time step
                for j in range(len(state)):
                    feed[model.initial_state[j]] = state[j]

                loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print('{}/{} (epoch {}), train_loss={:.3f}, time/batch={:.3f}'
                      .format(ep * data.n_batches + i, args.num_epochs * data.n_batches, ep, loss, end - start))

            # save model
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=ep * data.n_batches + i)
            print("model saved to {}".format(checkpoint_path))


class Loader:
    """Text data loader.

    Attributes:
        data_path (str):   Path to txt file containing training data
        batch_size (int)   Batch size when training RNN
        seq_length (int):  Maximum sequence length
        chars (tuple):     Tuple of all unique characters in text (upper case/lower case are treated differently)
        vocab (dict):      Dictionary mapping each character to a unique index (integer)
        vocab_size (int):  Number of unique characters
        embedding (array): Training data as sequence of integers
        n_batches (int):   Number of batches of training data
        x_batches (array): List of arrays of training features
        y_batches (array): List of arrays of training labels
        pointer (int):     Pointer to start of next batch of data
    """
    def __init__(self, data_path, batch_size, seq_length):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.chars = ()
        self.vocab = {}
        self.vocab_size = 0
        self.embedding = None

        self.n_batches = 0
        self.x_batches, self.y_batches = None, None
        self.pointer = 0

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        """pre-process data"""
        print('Pre-processing data...')
        with open(self.data_path, 'r') as f:
            data = f.read()
        counter = collections.Counter(data)                         # count occurences of each character in data
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # sort characters in descending order of frequency
        self.chars, _ = zip(*count_pairs)                           # all characters in descending order of frequency
        self.vocab_size = len(self.chars)                           # number of unique characters
        self.vocab = dict(zip(self.chars, range(len(self.chars))))  # dictionary mapping character to number
        self.embedding = np.array(list(map(self.vocab.get, data)))  # look up index of each character in training data

    def create_batches(self):
        """split data into training batches"""
        self.n_batches = int(self.embedding.size / (self.batch_size * self.seq_length))
        assert self.n_batches > 0, 'Not enough data. Make batch_size and/or seq_length smaller.'

        # truncate training data so it is equally divisible into batches
        self.embedding = self.embedding[:self.n_batches * self.batch_size * self.seq_length]
        x_train = self.embedding

        # y is the same as x, except shifted one character over (with wraparound)
        y_train = np.empty(x_train.shape)
        y_train[:-1] = x_train[1:]
        y_train[-1] = x_train[0]

        # split training data into equal sized batches
        self.x_batches = np.split(x_train.reshape(self.batch_size, -1), self.n_batches, 1)
        self.y_batches = np.split(y_train.reshape(self.batch_size, -1), self.n_batches, 1)

    def next_batch(self):
        """return current batch, increment pointer by 1 (modulo n_batches)"""
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.n_batches
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/trump.txt', help='file containing training data')
    parser.add_argument('--save_dir', type=str, default='../models/rnn/', help='directory to save models')
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=50, help='mini-batch size')
    parser.add_argument('--seq_length', type=int, default=50, help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for RMS prop')
    parser.add_argument('--load_from', type=str, default=None, help='path to saved model & configurations')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
