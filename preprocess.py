# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np
from config import ModelConfig


# TODO refactoring. split below
# 1. pad zero with max len is sr * max_duration
# 2. n_frames % batch_size = 0
def _resize_wav(wav, sr, duration):
    assert(wav.ndim <= 2)

    n_samples = sr * duration
    if wav.ndim == 1:
        pad_width = (0, n_samples - wav.shape[-1])
        wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        pad_width = ((0, 0), (0, n_samples - wav.shape[-1]))
        wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)

    unit = ModelConfig.L_FRAME * ModelConfig.BATCH_SIZE
    pad_len = 0
    if n_samples % unit > 0:
        pad_len = (unit - (n_samples % unit))

    if wav.ndim == 1:
        pad_width = (0, pad_len)
        wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)
        wav = wav[:wav.shape[-1] - ModelConfig.L_HOP]
    else:
        pad_width = ((0, 0), (0, pad_len))
        wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)
        wav = wav[:, :wav.shape[-1] - ModelConfig.L_HOP]
    return wav


# Batch considered
def get_mixed_wav(filenames, sr=ModelConfig.SR, sec=ModelConfig.MAX_SECONDS):
    return np.array(map(lambda f: _resize_wav(librosa.load(f, sr=sr, mono=True, duration=sec)[0], sr, sec), filenames))


# Batch considered
def get_src1_src2_wav(filenames, sr=ModelConfig.SR, sec=ModelConfig.MAX_SECONDS):
    wav = np.array(map(lambda f: _resize_wav(librosa.load(f, sr=sr, mono=False, duration=sec)[0], sr, sec), filenames))
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