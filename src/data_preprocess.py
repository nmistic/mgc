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
        signal = signal[(n_sample-n_sample_fit)/2:(n_sample + n_sample_fit)/2]import sys
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

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT,
                                                         n_mels=N_MELS) ** 2, ref=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        misc.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        print(melspect.shape)

    return melspect


if __name__ == '__main__':
    log_scale_melspectrogram(sys.argv[1], True)
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

    # mel scale is scale of pitches judged by listeners to be equidistant from one another
    # frequency measurement

    # convert power spectrogram (A^2) to dB units

    # deprecated ref_power
    # melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP,
    # n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT,
                                                         n_mels=N_MELS) ** 2, ref=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        misc.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        print(melspect.shape)

    return melspect


if __name__ == '__main__':
    log_scale_melspectrogram(sys.argv[1], True)
