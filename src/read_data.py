import os
import sys
import numpy as np
import pandas as pd
import data_preprocess as dp

labels_file = '../data/labels.csv'

# tags in order as they appear in labels.csv
# DO NOT CHANGE THIS ORDER
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# labels consists of (path, label) tuples
labels = pd.read_csv(labels_file)


# retrieve all label values as one hot vectors from labels.csv
def get_labels(labels_dense=labels['label'], num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# get dB values of all signals using the dp.log_scale_spectrogram function
def get_melspectrograms(labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([dp.log_scale_melspectrogram(i) for i in labels_dense['path']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms


# get dB values of signal at index using the dp.log_scale_spectrogram function
def get_melspectrograms_indexed(index, labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([dp.log_scale_melspectrogram(i) for i in labels_dense['path'][index]])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms

