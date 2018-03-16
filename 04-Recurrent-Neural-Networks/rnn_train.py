"""
Train the character-level RNN described in rnn_model.py
"""

import argparse
import tensorflow as tf


def train(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/', help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='../models/rnn/', help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50, help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for rmsprop')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
