import numpy as np
import tensorflow as tf
import time
import sys

train_file = sys.argv[1]
with open(train_file, 'r') as f:
    transcript = f.read()
transcript = transcript.replace('\n', ' ')
unique_chars = sorted(list(set(transcript)))
vocab_size = len(unique_chars)
char2idx = dict(zip(unique_chars, range(vocab_size)))


def pre_process(num_steps):
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
    return data_train, x_val, y_val, x_test, y_test


def build_graph(cell_type=None, state_size=100, num_classes=vocab_size, batch_size=32, num_steps=200, num_layers=3,
                learning_rate=1e-4):

    graph = {'batch_size': batch_size}
    tf.reset_default_graph()

    graph['x'] = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    graph['y'] = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, graph['x'])

    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    graph['init_state'] = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, graph['final_state'] = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=graph['init_state'])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b
    graph['predictions'] = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(graph['predictions'], 1), graph['y'])
    graph['accuracy'] = tf.reduce_mean(tf.cast(correct, tf.float32))

    graph['total_loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    graph['train_step'] = tf.train.AdamOptimizer(learning_rate).minimize(graph['total_loss'])
    graph['saver'] = tf.train.Saver()

    return graph


def main(num_steps):
    g = build_graph(cell_type='GRU', num_steps=num_steps)
    x, y, train_step, saver, batch_size, accuracy, saver = \
        g['x'], g['y'], g['train_step'], g['saver'], g['batch_size'], g['accuracy'], g['saver']

    data_train, x_val, y_val, x_test, y_test = pre_process(num_steps)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        print('Starting training...')
        start_time = time.time()
        best_accuracy = 0
        sess.run(init)

        for i in range(100):
            np.random.shuffle(data_train)
            for j in range(data_train.shape[0] // batch_size):
                start, end = j * batch_size, (j + 1) * batch_size
                x_batch, y_batch = data_train[start:end, :num_steps], data_train[start:end, num_steps:]
                sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
            val_accuracy = sess.run(accuracy, feed_dict={x: x_val, y: y_val})
            print('Validation accuracy:', val_accuracy)
            if val_accuracy > best_accuracy:
                saver.save(sess, '../models/rnn')
                print('Saving model...')
            else:
                saver.restore(sess, '../models/rnn')
                print('Previous accuracy: {}, restoring model...'.format(best_accuracy))

        print('Training complete in {:.2f}s'.format(time.time() - start_time))
        test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print('Best test accuracy: {:.2f}%'.format(100 * test_accuracy))
