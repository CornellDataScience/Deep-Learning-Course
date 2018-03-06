import tensorflow as tf
import numpy as np
import sys

save_path = "saved/rnn.ckpt"
alice_file = "alice_with_periods.npz"

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
alice_load = np.load(alice_file)
alice_embed = embed_output(alice_load['words'], alice_load['vocab'])  # Alice as sequence of ints, shape [num_words]
embedding_size = len(alice_load['vocab'])


# Split
sequence_length = 25
truncated_length = (len(alice_embed)//sequence_length) * sequence_length  # Need to make everything divisible
alice_embed = alice_embed[:truncated_length]
alice_split = np.reshape(alice_embed, (-1, sequence_length))  # break the text into fixed length sequences
num_sequences = alice_split.shape[0]
indices = np.random.permutation(num_sequences)
pct_train = 0.9
training_idx, test_idx = indices[:int(pct_train*num_sequences)], indices[int(pct_train*num_sequences):]
alice_train, alice_test = alice_split[training_idx, :], alice_split[test_idx, :]

num_units = 256  # number of RNN units in a RNNCell
num_layers = 3


# Do training
with tf.Graph().as_default():
    tf.set_random_seed(seed)

    x = tf.placeholder(shape=[None, sequence_length], dtype=tf.int32)  # Input sequence placeholder [batch, sequence]
    x_hot = tf.one_hot(x, depth=embedding_size)  # one-hot encode x
    batch_size_tensor = tf.shape(x)[0]  # This is a scalar tensor

    num_steps = sequence_length

    with tf.variable_scope('Unrolled') as scope:
        cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

        # The state of the RNN is the "zero state" at the start of every sequence. This is the initial state
        state = cell1.zero_state(batch_size=batch_size_tensor, dtype=tf.float32)

        # Unroll the graph num_steps back into the "past"
        outputs = []  # python list of tensors so we can keep track of each timestep
        for i in range(num_steps-1):  # need to unroll num_steps-1 units back in time
            if i > 0: scope.reuse_variables()  # Reuse the parameters created in the 1st RNN cell
            output, state = cell1(x_hot[:, i, :], state, scope='Cell1')  # Step the RNN through the sequence
            output, state = cell2(output, state, scope='Cell2')  # 2nd layer
            output, state = cell3(output, state, scope='Cell3')  # 3rd layer
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1, name='Outputs')

    with tf.variable_scope('Softmax'):
        w = tf.get_variable(
            name='Weight',
            initializer=tf.truncated_normal([cell3.output_size, embedding_size], stddev=0.01))
        b = tf.get_variable(name='Bias', initializer=tf.zeros(embedding_size))

        flattened = tf.reshape(outputs, (-1, cell3.output_size))  # Broadcasting doesn't work properly for tf.matmul
        matmul = tf.reshape(tf.matmul(flattened, w), shape=(-1, num_steps-1, embedding_size))
        scores = tf.add(matmul, b, name='Scores')
        softmax = tf.nn.softmax(scores, name='Softmax')

    # Shift over the inputs to create the labels
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=x[:, 1:]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session().as_default() as sess:
        num_epochs = sys.maxsize  # number of training epochs
        batch_size = 32

        training_size = alice_train.shape[0]

        saver = tf.train.Saver()

        try:
            saver.restore(sess, save_path)
            print("Restored Model!")
        except Exception:
            print("Initializing Model!")
            sess.run(tf.global_variables_initializer())

        try:
            num_batches = training_size // batch_size
            for epoch in range(num_epochs):
                perm = np.random.permutation(training_size)  # Every epoch, get a new set of batches
                avg_loss = 0
                for i in range(0, training_size, batch_size):
                    idx = perm[i:i + batch_size]  # Select indices for batch
                    x_batch = alice_train[idx]
                    _, batch_loss = sess.run([train_step, loss], feed_dict={x: x_batch})
                    avg_loss += batch_loss
                print("epoch %6d, loss=%6f" % (epoch + 1, avg_loss/num_batches))
        except SystemExit as e:
            print("Interrupted Training! Saving model to %s" % save_path)
            # Save model here
            saver.save(sess, save_path)
            print("Exiting...")
            exit()
        except KeyboardInterrupt as e:
            print("Interrupted Training!")

        # Save model here
        print("Saving model to %s" % save_path)
        saver.save(sess, save_path)
        sess.close()

    print("Proceeding to Inference...")
    tf.set_random_seed(seed)

    # Number of sentences to generate
    num_to_generate = tf.placeholder(tf.int32, shape=(), name='NumToGenerate')

    # Select a random word to start each sentence
    random_start = tf.random_uniform(shape=(num_to_generate,), maxval=embedding_size, dtype=tf.int32)

    # Reuse ALL variables
    with tf.variable_scope('Softmax', reuse=True):
        w = tf.get_variable(name='Weight')
        b = tf.get_variable(name='Bias')


    def do_softmax(tensor):
        """Helper function to compute softmax."""
        scores = tf.matmul(tensor, w) + b
        softmax = tf.nn.softmax(scores, name='Softmax')
        return softmax


    # Again, reuse ALL variables in this scope!
    with tf.variable_scope('Unrolled', reuse=True) as scope:
        cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

        # The state of the RNN is the "zero state" at the start of every sequence. This is the initial state.
        # Feed into `initial_state` with `feed_dict` if you want to use the state of a prior sequence
        initial_state = state = cell1.zero_state(batch_size=num_to_generate, dtype=tf.float32)

        # One-hot encode first word, and treat it as 1st output
        prev_word = tf.one_hot(random_start, depth=embedding_size)

        # Generate the sentence
        outputs = [prev_word]  # python list of tensors so we can keep track of all the outputs
        for i in range(num_steps-1):  # We already "made" the first word, so generate `num_steps-1` more
            output, state = cell1(prev_word, state, scope='Cell1')  # Step the RNN through the sequence
            output, state = cell2(output, state, scope='Cell2')  # 2nd layer
            output, state = cell3(output, state, scope='Cell3')  # 3rd layer
            prev_word = do_softmax(output)
            outputs.append(prev_word)

        final_state = state  # Useful if you want longer outputs
        outputs = tf.stack(outputs, axis=1, name='Outputs')

    generated = tf.argmax(outputs, axis=-1, name='Generated')

    with tf.Session().as_default() as sess:
        if save_path is not None:
            saver.restore(sess, save_path)
        else:
            sess.run(tf.global_variables_initializer())
        generated = sess.run(generated, feed_dict={num_to_generate: 10})
        sentences = [[alice_load['vocab'][embedding] for embedding in sentence] for sentence in generated]
        for sentence in sentences:
            print(sentence)



