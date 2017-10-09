from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np

from time import time, strftime


class RNN(object):
    """This class implements a Recurrent Neural Network (RNN) in TensorFlow.

    This particular type of RNN is called a Long Short-Term Memory (LSTM). It helps mitigate the effects of the
    Vanishing/Exploding Gradient problem by managing its internal cell state with forget, input, and output gates. The
    actual LSTM implementation is handled by TensorFlow with the `BasicLSTMCell` class.
    """

    def __init__(self, embedding_size, num_steps, cell_size=128, seed=None, load_model=None):
        """Initializes the architecture of the RNN and returns an instance.
        Args:
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            num_steps:      An integer that is the number of unrolled steps that the RNN takes. No provided sequence
                            should be longer than this number. This number is related to the ability of the RNN to
                            understand long-term dependencies in the data.
            cell_size:      An integer that is equal to the size of the LSTM cell. This is directly related to the
                            state size and number of parameters of the cell.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._embedding_size = embedding_size
        self._seed = seed
        self._num_steps = num_steps  # Tuples are used to ensure the dimensions are immutable
        self._cell_size = cell_size
        self._num_steps = num_steps

        self._last_time = 0  # Used by train to keep track of time
        self._iter_count = 0  # Used by train to keep track of iterations
        self._needs_update = False  # Used by train to indicate when enough time has passed to print info about progress

        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)  # ensure TensorFlow uses our random seed

            # In this block of code, we define the inputs to the network and make the input data one-hot encoded
            with tf.variable_scope('Inputs'):

                # Here we define the shape of the various input tensors. `None` means that dimension have have any size
                x_shape = [None, num_steps]  # [Batch dimension, time dimension]
                y_shape = [None]  # [Batch dimension]
                lengths_shape = [None]  # [Batch dimension]

                # Placeholder tensors for data
                self._x = tf.placeholder(tf.int32, shape=x_shape, name='X')  # Inputs
                self._y = tf.placeholder(tf.int32, shape=y_shape, name='Y')  # Desired Outputs

                # We also need to know the lengths of each X because they can vary in length, up to `num_steps`
                self._lengths = tf.placeholder(tf.int32, shape=lengths_shape, name='Lengths')

                # Get the actual batch size. We don't know this beforehand because we set the batch dimension as `None`
                self._batch_size = tf.shape(self._x)[0]  # This is a scalar tensor (tensor of a single number)

                # X as one-hot encoded. Hence the tensor is now of shape [batch_size, num_steps, embedding_size]
                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')

            with tf.variable_scope('Unrolled') as scope:
                # We need to tell TensorFlow what type of RNN cell structure to use. Here we define it as an LSTM
                rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=cell_size)

                # The state of the RNN is the "zero state" at the start of every sequence. This is the initial state
                state = rnn_cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)

                # Unroll the graph num_steps back into the "past"
                self._outputs = []  # python list of tensors so we can keep track of each timestep
                for i in range(num_steps):  # need to unroll up to num_steps units back in time
                    if i > 0: scope.reuse_variables()  # Reuse the parameters created in the 1st RNN cell
                    output, state = rnn_cell(self._hot[:, i, :], state)  # Step the RNN through the sequence
                    self._outputs.append(output)

                final_output = self._outputs[-1]  # Get the last output as the final result of the RNN
                with tf.variable_scope('Softmax'):
                    # Parameters
                    w = tf.get_variable(
                        name='Weights',
                        initializer=tf.truncated_normal([rnn_cell.output_size, embedding_size], stddev=0.1))

                    b = tf.get_variable(
                        name='Bias',
                        initializer=tf.truncated_normal([embedding_size], stddev=0.1))

                    scores = tf.matmul(final_output, w) + b  # The raw class scores to be fed into the loss function

                    self._y_hat = tf.nn.softmax(
                        scores, name='Y-Hat')  # Class probabilities, [batch_size, embedding_size]

                    # Get the label of the most likely class. Yields a vector of predicted labels of shape [batch_size,]
                    self._predictions = tf.argmax(self._y_hat, axis=1, name='Predictions')

                with tf.variable_scope('Optimization'):
                    self._loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(  # Cross-entropy loss
                            # This function expects the raw scores instead of class probabilities as input to avoid
                            # errors that happen with decimal numbers. The end result is the same as what we learned
                            logits=scores,
                            labels=self._y),
                        name='Loss')
                    self._train_step = tf.train.AdamOptimizer().minimize(self._loss)  # Optimizer

            self._sess = tf.Session()  # create the session so we can prepare to run the model

            # Either load a previous model into the session or initialize the parameters for a new session
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    load_model = os.path.abspath(load_model)
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, y_train, lengths_train=None, num_epochs=1000,  batch_size=None, progress_info=True,
              progress_interval=5):
        """Trains the model on a given dataset.
        Because the loss is averaged over batch, a larger batch size will result in a more stable loss function with
        potentially better results when applying the model, although having a smaller batch size means less memory
        consumption and usually faster training per epoch.
        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [num_steps, batch_size] if _time_major=True, otherwise [batch_size, num_steps]. Each element of this
                matrix should be the index of the item being trained on in its one-hot encoded representation. Indices
                are used instead of the full one-hot vector for efficiency.
            y_train:  A numpy ndarray that contains the labels that correspond to the data being trained on. Should have
                a shape of [batch_size]. Each element is the index of the on-hot encoded representation of the label.
            lengths_train:  A numpy ndarray that contains the sequence length for each element in the batch. If none,
                sequence lengths are assumed to be the full length of the time dim.
            num_epochs:  The number of iterations over the provided batch to perform until training is considered to be
                complete. If all your data fits in memory and you don't need to mini-batch, then this should be a large
                number (>=1000). Otherwise keep this small (<50) so the model doesn't become skewed by the small size of
                the provided mini-batch too quickly. It is expected that the code that selects the batch size will
                call this train method once with each new batch (or just once if mini-batching is not necessary).
            batch_size:  The size of the mini-batch to use during training. If None, uses entire dataset as the batch.
            progress_info:  If true, print what the current loss and percent completion over the course of training.
            progress_interval:  How many seconds to wait to print the next progress update, if `progress_info` is True.

        Returns:
            The loss value after training.
        """
        with self._sess.as_default():

            if progress_info: print("Starting training for %d epochs" % num_epochs)

            fetches = (self._train_step, self._loss)  # These are the tensors/operations we want TensorFlow to perform

            num_samples = x_train.shape[0]
            if num_samples is not y_train.shape[0] or num_samples is not lengths_train.shape[0]:
                raise ValueError("Sample size is not consistent. Ensure all inputs have same number of samples")

            if batch_size is None: batch_size = num_samples
            if batch_size > num_samples:
                raise ValueError("Batch size cannot exceed the number of samples!")

            # Training loop for the given batch
            for epoch in range(num_epochs):

                shuffle = np.random.permutation(num_samples)  # Each epoch, randomly shuffle indices
                for i in range(0, num_samples, batch_size):
                    indices = shuffle[i:i+batch_size]  # Indices to use for batch
                    x_batch = x_train[indices]
                    y_batch = y_train[indices]
                    lengths_batch = lengths_train[indices]

                    # Feed the data into the graph and run one training step. Returned are the results of `fetches`
                    _, loss_val = self._sess.run(
                        fetches,
                        feed_dict={self._x: x_batch, self._y: y_batch, self._lengths: lengths_batch})

                    current_time = time()
                    if (current_time - self._last_time) >= progress_interval:  # Print progress every few seconds
                        self._last_time = current_time
                        self._needs_update = True
                    else:
                        self._needs_update = False

                    if progress_info and self._needs_update:  # Only print progress when needed
                        print("Current Loss Value: %.10f, Percent Complete: %.4f"
                              % (loss_val, (epoch*num_samples+i) / (num_epochs*num_samples) * 100))

                    self._iter_count += 1

            if progress_info: print("Completed Training.")
        return loss_val

    def apply(self, x_data, lengths_data=None):
        """Applies the model to the batch of data provided. Typically called after the model is trained.
        Args:
            x_data:  An ndarray of the data to apply the model to. Should have a similar shape to the training data.
                The time dimension may be smaller than `num_steps` if it makes sense within the context of the problem.
                If `time_major` varies from what was supplied in the constructor, then the data will be transposed
                before being fed to the model. For efficiency, it is therefore best to have data in the same format as
                the model.
            lengths_data:  An optional numpy ndarray that describes the sequence length of each element in the batch.
                Should be a vector the length of the batch size. If None, then sequence length is assumed to be the same
                for each batch element and will be the length of the time dimension. If it is None, then the size of the
                time dimension of `x_data` will be used.

        Returns:
            A numpy ndarray of the data, with shape [batch_size, embedding_size]. Rows are class probabilities.
            Example: result.shape is [batch_size, 100] when there are 100 unique words in the chosen dictionary.
        """
        with self._sess.as_default():
            if lengths_data is None:
                return self._sess.run(self._y_hat, feed_dict={self._x: x_data})
            else:
                return self._sess.run(self._y_hat, feed_dict={self._x: x_data, self._lengths: lengths_data})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.
        Args:
            save_path:  The relative path to the file. By default, it is
                saved/LSTM-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/LSTM-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)