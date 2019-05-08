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
