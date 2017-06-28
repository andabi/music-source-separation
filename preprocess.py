# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa.display
import numpy as np

sr = 16000
len_frame = 4096
len_hop = len_frame


def get_mixed_wav(filenames, sr=sr):
    return np.array(map(lambda f: librosa.load(f, sr=sr, mono=True)[0], filenames))


def get_src1_src2_wav(filenames, sr=sr):
    return np.array(map(lambda f: librosa.load(f, sr=sr, mono=False)[0], filenames))


def to_spectogram(wav, len_frame=len_frame, len_hop=len_hop):
    return np.array(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav))


def to_wav(stft_maxrix, len_hop=len_hop):
    return np.array(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix))


def write(wav, filename, sr=sr):
    librosa.output.write_wav(filename, wav, sr)


def get_amplitude(stft_matrix):
    return np.abs(stft_matrix)


def get_phase(stft_maxtrix):
    return np.angle(stft_maxtrix)