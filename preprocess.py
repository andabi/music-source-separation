# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np

sr = 16000
len_frame = 4096
len_hop = len_frame


# Batch considered
def get_mixed_wav(filenames, sr=sr):
    return np.array(map(lambda f: librosa.load(f, sr=sr, mono=True)[0], filenames))


# Batch considered
def get_src1_src2_wav(filenames, sr=sr):
    wav = np.array(map(lambda f: librosa.load(f, sr=sr, mono=False)[0], filenames))
    return wav[:, 0], wav[:, 1]


# Batch considered
def to_spectogram(wav, len_frame=len_frame, len_hop=len_hop):
    return np.array(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav))


# Batch considered
def to_wav(stft_maxrix, len_hop=len_hop):
    return np.array(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix))


# Batch considered
def write_wav(wav, filenames, sr=sr):
    pair = zip(wav, filenames)
    map(lambda p: librosa.output.write_wav(p[1], p[0], sr), pair)


def get_amplitude(stft_matrix):
    return np.abs(stft_matrix)


def get_phase(stft_maxtrix):
    return np.angle(stft_maxtrix)