import os

import tensorflow as tf
import random
from time import strftime
from time import time


class LogisticClassifier:

    def __init__(self, x_train, y_train, num_epochs=10000, beta=0.01, seed=None, load_model=None):
        self.__seed = random.randint(0, 2**30) if seed is None else seed
        self.__beta = beta
        self.__graph = tf.Graph()
        num_features = x_train.shape[1]
        with self.__graph.as_default():
            # SETUP THE MODEL
            # Symbolic Vars
            self.__X = tf.placeholder(tf.float32, (None, num_features), name="Features")
            self.__Y = tf.placeholder(tf.int32, (None,), name="Targets")

            # Parameters
            w = tf.Variable(tf.random_normal((num_features, 2), stddev=0.1, dtype=tf.float32, seed=self.__seed, name="Parameters"))
            b = tf.Variable(tf.random_normal((2,), stddev=0.1, dtype=tf.float32, seed=self.__seed, name="Bias"))

            # Output Function (before sigmoid, passed to the loss function)
            y = tf.add(tf.matmul(self.__X, w), b)
            #y = tf.matmul(self.__X, w)

            # Output Function (after sigmoid, represents the actual predicted probabilities)
            self.__y_hat = tf.nn.softmax(y, name="Y-hat")

            # Cost and Loss functions
            regularizer = tf.nn.l2_loss(w, name="Regularizer")  # Penalize parameters on their L2 norm squared
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.__Y, logits=y),  # use cross entropy as loss
                name="Loss"
            )
            cost = tf.add(self.__beta * regularizer, loss, name="Cost")  # The final cost function to minimize

            # Optimizer setup
            train_step = tf.train.AdamOptimizer().minimize(cost)

            # Summary data
            match_results = tf.equal(tf.cast(tf.argmax(self.__y_hat, axis=1), dtype=tf.int32), self.__Y)  # Vector of bool representing success of predictions
            self.__accuracy = tf.reduce_mean(
                tf.cast(match_results, dtype=tf.float32),  # scalar of percentage of correct predictions
                name="Accuracy"
            )
            # Setup TensorFlow Session and initialize graph variables
            self.sess = tf.Session(graph=self.__graph)

            with self.sess.as_default():
                self.__saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    self.__saver.restore(self.sess, load_model)
                    print("Model Restored!")
                else:
                    self.sess.run(tf.global_variables_initializer())

                # Training loop for parameter tuning
                print("Starting training for %d epochs" % num_epochs)
                last_time = time()
                for epoch in range(num_epochs):
                    _, cost_val = self.sess.run([train_step, cost], feed_dict={self.__X: x_train, self.__Y: y_train})
                    current_time = time()
                    if (current_time-last_time) >= 5:
                        last_time = current_time
                        print("Current Cost Value: %.10f, Percent Complete: %f" % (cost_val, epoch/num_epochs))
                        '''the_y, the_y_hat = self.sess.run([y, self.__y_hat], feed_dict={self.__X: x_train, self.__Y: y_train})
                        print(the_y)
                        print(the_y_hat)'''
                print("Completed Training.")

                # Training Summary
                training_accuracy = self.sess.run(self.__accuracy, feed_dict={self.__X: x_train, self.__Y: y_train})
                print("Training Accuracy: %f" % training_accuracy)

    def predict_values(self, x_data):
        with self.sess.as_default():
            return self.sess.run(self.__y_hat, feed_dict={self.__X: x_data})

    def test_accuracy(self, x_data, y_data):
        with self.sess.as_default():
            return self.sess.run(self.__accuracy, feed_dict={self.__X: x_data, self.__Y: y_data})

    def save_model(self, save_path=None):
        with self.sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "ryan/saved-networks/LogisticClassifier-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            path = self.__saver.save(self.sess, save_path)
            print("Model successfully saved in file: %s" % path)