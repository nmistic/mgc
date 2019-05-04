import os
import sys
import numpy as np
import pandas as pd
import data_preprocess as dp

labels_file = '../data/labels.csv'

tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

labels = pd.read_csv(labels_file)
