# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np
from config import *


# Batch considered
def get_mixed_wav(filenames, sr=SR):
    return np.array(map(lambda f: librosa.load(f, sr=sr, mono=True)[0], filenames))


# Batch considered
def get_src1_src2_wav(filenames, sr=SR):
    wav = np.array(map(lambda f: librosa.load(f, sr=sr, mono=False)[0], filenames))
    return wav[:, 0], wav[:, 1]


# Batch considered
def to_spectrogram(wav, len_frame=L_FRAME, len_hop=L_HOP):
    return np.array(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav))


# Batch considered
def to_wav(stft_maxrix, len_hop=L_HOP):
    return np.array(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix))


# Batch considered
def write_wav(wav, filenames, sr=SR):
    pair = zip(wav, filenames)
    map(lambda p: librosa.output.write_wav(p[1], p[0], sr), pair)


# Batch considered
def get_magnitude(stft_matrixes):
    return np.abs(stft_matrixes)


# Batch considered
def get_phase(stft_maxtrixes):
    return np.angle(stft_maxtrixes)


# Batch considered
def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1j * phases)


# Batch considered
def time_freq_mask(target_src, remaining_src):
    # shape = (B, T, F)
    with np.errstate(divide='ignore'):
        denominator = (np.abs(target_src) + np.abs(remaining_src))
        denominator[denominator == 0] = float('inf')
        mask = np.abs(target_src) / denominator
    return mask