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
