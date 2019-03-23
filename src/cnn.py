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