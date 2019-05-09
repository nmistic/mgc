import sys
import numpy as np
# music and audio analysis package librosa
import librosa as lb
from scipy import misc

# 12 kHz
Fs         = 12000
# length of the fft window
N_FFT      = 512
# extra argument - number of mels
N_MELS     = 96
# number of samples between successive frames
N_OVERLAP  = 256
DURA       = 29.12


def log_scale_melspectrogram(path, plot=False):
    # load song signal as waveform and sampling rate from path
    signal, sr = lb.load(path, sr=Fs)

    n_sample = signal.shape[0]
    n_sample_fit = int(DURA * Fs)\

    # does the sample fit ?
    if n_sample < n_sample_fit:
        # if smaller, fill the remaining with zeros
        signal = np.hstack((signal, np.zeros((int(DURA * Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        # if larger, take the middle part from (n_sample-n_sample_fit)/2 to (n_sample+n_sample_fit)/2
        signal = signal[(n_sample-n_sample_fit)/2:(n_sample + n_sample_fit)/2]