import tensorflow as tf
import random
import numpy as np
import time
from tensorflow.contrib.layers import flatten
import Data_DH_2

seq_length = 1024
line_num = 1000

# Data
X_data, y_data, Label = Data_DH_2.data_read(seq_length, line_num)
X_data, y_data = Data_DH_2.data_embedding(X_data, y_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Shuffle
Data = list(zip(X_data, y_data, Label))
random.shuffle(Data)
X_data, y_data, Label = zip(*Data)
X_data, y_data, Label = np.array(X_data), np.array(y_data), np.array(Label)

# Data split
X_train, y_train = X_data[0:int(len(X_data)*0.9)-1], y_data[0:int(len(y_data)*0.9)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.9):int(len(X_data)*0.95)-1], y_data[int(len(X_data)*0.9):int(len(X_data)*0.95)-1]
X_test, y_test, Label_test = X_data[int(len(X_data)*0.95):len(X_data)-1], y_data[int(len(X_data)*0.95):len(y_data)-1], Label[int(len(X_data)*0.95):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)), "Valuate data volume: {}".format(len(X_valuate)), "Test data volume: {}".format(len(X_test)))

# Hyper-parameters
batch_size = 128
lr = 0.0001
hidden_units = seq_length / 8
maxlen = 8
num_blocks = 3
num_epochs = 2
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

# Modules
def ln(inputs, epsilon=1e-8, scope="ln"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0.5, training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(outputs, Q, K, type="query")

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)

    return outputs

def mask(inputs, queries=None, keys=None, type=None):

    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])

        # Apply masks to inputs
        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0.5,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):

    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model)
        K = tf.layers.dense(keys, d_model)
        V = tf.layers.dense(values, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs

def feedforward(inputs, num_units, scope="multihead_attention", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs

def positional_encoding(inputs, maxlen, masking=True, scope="positional_encoding"):
    E = inputs.get_shape().as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def encoder_attention_block(inputs, keep_prop):
    enc = multihead_attention(queries=inputs,
                              keys=inputs,
                              values=inputs,
                              dropout_rate=keep_prop)
    enc = feedforward(enc, num_units=[4 * hidden_units, hidden_units])

    return enc

def decoder_attention_block(input1, input2, keep_prop):
    memory = multihead_attention(queries=input1,
                              keys=input1,
                              values=input1,
                              causality=True,
                              dropout_rate=keep_prop)

    dec = multihead_attention(queries=memory,
                              keys=input2,
                              values=input2,
                              dropout_rate=keep_prop)

    dec = feedforward(dec, num_units=[4 * hidden_units, hidden_units])

    return dec

def linear(inputs):
    fc_W = tf.Variable(tf.truncated_normal(shape=(int(seq_length / 8), int(seq_length / 8)), mean=0, stddev=0.1))
    logits = tf.einsum('ntd,dk->ntk', inputs, fc_W)
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")

    reshaped_output = output_scale_factor * logits

    return reshaped_output

x = tf.placeholder(tf.float32, [None, maxlen, hidden_units])
y = tf.placeholder(tf.float32, [None, maxlen, hidden_units])
keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

time_start = time.time()
# Model
with tf.variable_scope("enc_num_blocks_1"):
    enc1 = encoder_attention_block(x, keep_prob_ph)
with tf.variable_scope("enc_num_blocks_2"):
    enc2 = encoder_attention_block(enc1, keep_prob_ph)
with tf.variable_scope("enc_num_blocks_3"):
    enc3 = encoder_attention_block(enc2, keep_prob_ph)

with tf.variable_scope("dec_num_blocks_1"):
    dec1 = decoder_attention_block(y, enc3, keep_prob_ph)
with tf.variable_scope("dec_num_blocks_2"):
    dec2 = decoder_attention_block(dec1, enc3, keep_prob_ph)
with tf.variable_scope("dec_num_blocks_3"):
    dec3 = decoder_attention_block(dec2, enc3, keep_prob_ph)

pred = linear(dec3)
l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss_operation = tf.sqrt(tf.reduce_mean(tf.nn.l2_loss(y - pred))) + l2
mean_absolute_error = tf.reduce_mean(tf.abs(y - pred)) + l2
training_operation = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_operation)

saver = tf.train.Saver()

test_losses, test_MAEs = [], []
valuate_losses, valuate_MAEs = [], []
train_losses, train_MAEs = [], []
train_time, val_time, test_time = [], [], []

def evaluate(X_data, y_data):
    sess = tf.get_default_session()
    batch_x, batch_y = X_data, y_data
    loss, mae = sess.run(
        [loss_operation, mean_absolute_error],
        feed_dict={
            x: batch_x,
            y: batch_y,
            keep_prob_ph: dropout_rate
        }
    )
    total_loss = np.mean(np.sum(loss))
    total_mae = np.mean(np.sum(mae))

    return total_loss, total_mae

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(num_epochs):
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, loss, mae = sess.run(
                [training_operation, loss_operation, mean_absolute_error],
                feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob_ph: dropout_rate
                }
            )
            train_losses.append(loss)
            train_MAEs.append(mae)
        train_losses.pop()
        validation_time_start = time.time()
        valuate_loss, valuate_mae = evaluate(X_valuate, y_valuate)
        validation_time_end = time.time()
        val_time.append(validation_time_end-validation_time_start)
        valuate_losses.append(valuate_loss)
        valuate_MAEs.append(valuate_mae)
        print("EPOCH {} ...".format(i + 1))
        print("Valuate RMSE = {:.4f}".format(valuate_loss), "Valuate MAE = {:.4f}".format(valuate_mae),
               "Validation time = {:.3f}".format(validation_time_end-validation_time_start))
        print()

    saver.save(sess, './DH2-{}'.format(seq_length))
    print("Model saved")
time_end = time.time()
train_time.append(time_end - time_start)
print("The time consumption of training stage = {:.3f}".format(time_end - time_start))

xt, yt, pt = [], [], []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    test_time_start = time.time()
    loss, mae, pred = sess.run(
        [loss_operation, mean_absolute_error, pred],
        feed_dict={
            x: X_test,
            y: y_test,
            keep_prob_ph: dropout_rate
        }
    )
    test_time_end = time.time()
    test_time.append(test_time_end - test_time_start)
    test_losses.append(loss)
    test_MAEs.append(mae)

    print("The final loss = {:.4f}".format(loss), "The final loss = {:.4f}".format(mae))
    print("The time consumption of test = {:.3f}".format(test_time_end - test_time_start))
