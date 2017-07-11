# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import librosa
import numpy as np
from config import ModelConfig


def _pad_wav(wav, sr, duration):
    assert(wav.ndim <= 2)

    n_samples = sr * duration
    if wav.ndim == 1:
        pad_width = (0, n_samples - wav.shape[-1])
    else:
        pad_width = ((0, 0), (0, n_samples - wav.shape[-1]))
    wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)

    return wav


# Batch considered
def get_mixed_wav(filenames, sec, sr=ModelConfig.SR):
    return np.array(map(lambda f: _pad_wav(librosa.load(f, sr=sr, mono=True, duration=sec)[0], sr, sec), filenames))


# Batch considered
def get_src1_src2_wav(filenames, sec, sr=ModelConfig.SR):
    wav = np.array(map(lambda f: _pad_wav(librosa.load(f, sr=sr, mono=False, duration=sec)[0], sr, sec), filenames))
    return wav[:, 0], wav[:, 1]


# Batch considered
def to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav))


# Batch considered
def to_wav(stft_maxrix, len_hop=ModelConfig.L_HOP):
    return np.array(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix))


# Batch considered
def write_wav(wav, filenames, sr=ModelConfig.SR):
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
    mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
    return mask