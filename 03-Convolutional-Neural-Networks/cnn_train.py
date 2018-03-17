"""
Train a CNN on the CIFAR-10 dataset.
Adapted from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10
"""

import argparse
import os
import numpy as np
import tarfile
import tensorflow as tf
import time
import sys
from six.moves import urllib
from cnn_model import CNN

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
TRAIN_FILES = ['../data/cifar-10-batches-bin/data_batch_%d.bin' % i for i in np.arange(1, 6, dtype=int)]
TEST_FILES = ['../data/cifar-10-batches-bin/test_batch.bin']


def train(args):
    """Train model"""
    data = CIFAR10(args.batch_size, TRAIN_FILES)

    # create save directory if it does not already exist
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print('Initializing model...')
    images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'input_images')
    distorted = distort_images(images)
    model = CNN(distorted)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print('Starting training...')
        for n in range(args.num_epochs):

            for i in range(data.n_batches):
                start = time.time()
                x, y = data.next_batch()
                loss, _ = sess.run([model.loss, model.train_step], feed_dict={images: x, model.labels: y})
                end = time.time()
                print('{}/{} (epoch {}), train_loss={:.3f}, time/batch={:.3f}'
                      .format(n * data.n_batches + i, args.num_epochs * data.n_batches, n, loss, end - start))

            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=n * data.n_batches)
            print("model saved to {}".format(checkpoint_path))


class CIFAR10:
    """Object representing dataset"""
    def __init__(self, batch_size, files):
        self.files = files
        self.batch_size = batch_size
        self.images, self.labels = None, None

        self.n_batches = 0
        self.x_batches, self.y_batches = None, None
        self.pointer = 0

        self.pre_process()
        self.create_batches()

    def pre_process(self):
        """Load and pre-process data"""

        print('Pre-processing data...')
        batches = ()
        binary = tf.placeholder(tf.string)
        for file in self.files:
            with open(file, 'rb') as f:
                byte_string = f.read()
            decoded = tf.decode_raw(binary, tf.uint8)
            with tf.Session() as sess:
                vectors = sess.run(decoded, feed_dict={binary: byte_string})
                batches += (np.reshape(vectors, [-1, 3073]), )
        data = np.vstack(batches)
        self.images = data[:, 1:].reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
        self.labels = data[:, 0]

    def create_batches(self):
        """Split data into training mini-batches"""
        self.n_batches = int(self.labels.size / self.batch_size)
        self.x_batches = np.array_split(self.images, self.n_batches)
        self.y_batches = np.array_split(self.labels, self.n_batches)

    def next_batch(self):
        """Return current batch, increment pointer by 1 (modulo n_batches)"""
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.n_batches
        return x, y


def distort_images(images):
    """Randomly distort a batch of images"""
    with tf.variable_scope('data_augmentation'):
        # Randomly crop a [height, width] section of the image
        distorted = tf.map_fn(lambda img: tf.random_crop(img, [24, 24, 3]), images)
        # Randomly flip the image horizontally
        distorted = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), distorted)
        # Subtract off the mean and divide by the variance of the pixels
        normalized = tf.map_fn(lambda img: tf.image.per_image_standardization(img), distorted)
        return normalized


def download_cifar(save_dir):
    """Download and extract CIFAR-10 dataset"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, (count * block_size) / total_size * 100))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        stat_info = os.stat(filepath)
        print('\nSuccessfully downloaded {} {} bytes.'.format(filename, stat_info.st_size))
    extracted_dir_path = os.path.join(save_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(save_dir)


def parse_arguments(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/', help='path to save/load CIFAR-10 data to/from.')
    parser.add_argument('--save_dir', type=str, default='../models/cnn/', help='directory to save trained models.')
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size.')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    download_cifar(args.data_dir)
    train(args)
