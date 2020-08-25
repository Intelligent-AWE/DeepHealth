import tensorflow as tf
import random
import numpy as np
import time
from tensorflow.contrib.layers import flatten
import Data_DH_1

seq_length = 1024
line_num = 1000

# Data
X_data, y_data = Data_DH_1.data_read(seq_length, line_num)
X_data = Data_DH_1.data_embedding(X_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Shuffle
Data = list(zip(X_data, y_data))
random.shuffle(Data)
X_data, y_data = zip(*Data)
X_data, y_data = np.array(X_data), np.array(y_data)

# Data split
X_train, y_train = X_data[0:int(len(X_data)*0.7)-1], y_data[0:int(len(y_data)*0.7)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1], y_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1]
X_test, y_test = X_data[int(len(X_data)*0.9):len(X_data)-1], y_data[int(len(X_data)*0.9):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)), "Valuate data volume: {}".format(len(X_valuate)), "Teat data volume: {}".format(len(X_test)))

# Hyper-parameters
batch_size = 128
lr = 0.0001
hidden_units = seq_length / 8
maxlen = 8
num_blocks = 3
num_epochs = 300
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

# Modules
def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=dropout_rate,
                        is_training=True, causality=False, scope="multihead_attention", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        print(Q.shape, K.shape, V.shape)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        print(Q_.shape, K_.shape, V_.shape)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        print(outputs.shape)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        print(outputs.shape)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)

    return outputs

def feedforward(inputs, num_units, scope="multihead_attention", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        print(outputs.shape)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        print(outputs.shape)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

def attention_block(inputs):
    enc = multihead_attention(queries=inputs,
                              keys=inputs,
                              num_units=hidden_units,
                              num_heads=num_heads,
                              dropout_rate=0.1,
                              is_training=True,
                              causality=False)
    enc = feedforward(enc, num_units=[4 * hidden_units, hidden_units])

    return enc

def linear(seq_len, inputs):
    logits = flatten(inputs)
    fc_W = tf.Variable(tf.truncated_normal(shape=(seq_len, 6), mean=0, stddev=0.1))
    fc_b = tf.Variable(tf.zeros(6))
    logits = tf.matmul(logits, fc_W) + fc_b

    return logits

def one_hot_encoding(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = 6
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

x = tf.placeholder(tf.float32, [None, maxlen, hidden_units])
y = tf.placeholder(tf.int32, [None, 6])

time_start = time.time()

# Blocks
with tf.variable_scope("num_blocks_1"):
    enc1 = attention_block(x)
with tf.variable_scope("num_blocks_2"):
    enc2 = attention_block(enc1)
with tf.variable_scope("num_blocks_3"):
    enc3 = attention_block(enc2)
pred = linear(seq_length, enc3)

l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
loss_operation = tf.reduce_mean(cross_entropy) + l2
training_operation = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

test_losses = []
test_accuracies = []
valuate_accuracies = []
valuate_losses = []
train_losses = []
train_accuracies = []
confusion_matrixes = []
train_time, val_time, test_time = [], [], []

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        loss, acc = sess.run(
            [loss_operation, accuracy_operation],
            feed_dict={
                x: batch_x,
                y: batch_y,
            }
        )
        total_accuracy += (acc * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(num_epochs):
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, loss, acc = sess.run(
                [training_operation, loss_operation, accuracy_operation],
                feed_dict={
                    x: batch_x,
                    y: one_hot_encoding(batch_y)
                }
            )
            train_accuracies.append(acc)
            train_losses.append(loss)
        validation_time_start = time.time()
        valuate_accuracy, valuate_loss = evaluate(X_valuate, one_hot_encoding(y_valuate))
        validation_time_end = time.time()
        val_time.append(validation_time_end-validation_time_start)
        valuate_accuracies.append(valuate_accuracy)
        valuate_losses.append(valuate_loss)
        print("EPOCH {} ...".format(i + 1))
        print("Valuate Accuracy = {:.4f}".format(valuate_accuracy), "Valuate Loss = {:.4f}".format(valuate_loss), "Validation time = {:.3f}".format(validation_time_end-validation_time_start))
        print()

    saver.save(sess, './DH1-{}'.format(seq_length))
    print("Model saved")
time_end = time.time()
train_time.append(time_end-time_start)
print("The time consumption of training stage = {:.3f}".format(time_end-time_start))

precision, recall, f1_score = [], [], []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    test_time_start = time.time()
    one_hot_prediction, loss, final_acc = sess.run(
        [pred, loss_operation, accuracy_operation],
        feed_dict={
            x: X_test,
            y: one_hot_encoding(y_test),
        }
    )
    test_time_end = time.time()
    test_time.append(test_time_end - test_time_start)
    test_accuracies.append(final_acc)
    test_losses.append(loss)
    print("The Final Test Accuracy = {:.5f}".format(final_acc))
    print("The time consumption of test = {:.3f}".format(test_time_end - test_time_start))