import sys
import pickle as pk
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
import read_data as rd

batch_size    = 4
learning_rate = 0.003
n_epoch       = 50
n_samples     = 1000                              # change to 1000 for entire dataset
cv_split      = 0.8                             
train_size    = int(n_samples * cv_split)                               
test_size     = n_samples - train_size


# initialize the weights of the layer
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# init the biases of a layer
def init_biases(shape):
    return tf.Variable(tf.zeros(shape))


def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def cnn(melspectrogram, weights, phase_train):
    # reshape the melspectrogram, bn it and reshape it
    x = tf.reshape(melspectrogram, [-1, 1, 96, 1366])
    x = batch_norm(melspectrogram, 1366, phase_train)
    x = tf.reshape(melspectrogram, [-1, 96, 1366, 1])

    # create the convolution layers with relu activation with same padding,
    # pooling layers with valid padding and dropouts
    # stride = 1
    # layer 1
    conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1],
                                  padding='SAME'), weights['bconv1'])
    conv2_1 = tf.nn.relu(batch_norm(conv2_1, 32, phase_train))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    # layer 2
    conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1],
                                  padding='SAME'), weights['bconv2'])
    conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    # layer 3
    conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1],
                                  padding='SAME'), weights['bconv3'])
    conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    # layer 4
    conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1],
                                  padding='SAME'), weights['bconv4'])
    conv2_4 = tf.nn.relu(batch_norm(conv2_4, 192, phase_train))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    # layer 5
    conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1],
                                  padding='SAME'), weights['bconv5'])
    conv2_5 = tf.nn.relu(batch_norm(conv2_5, 256, phase_train))
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    # flatten out the output layer, with sigmoid activation
    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))

    return p_y_X
