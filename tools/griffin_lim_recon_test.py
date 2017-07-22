# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import librosa
import numpy as np
from preprocess import griffin_lim

file_path = '../dataset/kpop/I LUV IT.wav'
sr = 16000
len_frame = 1024
len_hop = len_frame / 4
num_iters = 50

wav, sr = librosa.load(file_path, sr=sr, mono=True)
spec = librosa.stft(wav, n_fft=len_frame, hop_length=len_hop)
mag, phase = np.abs(spec), np.angle(spec)

# Griffin Lim reconstruction
wav_recon = griffin_lim(mag, len_frame, len_hop, num_iters=num_iters, length=wav.shape[0])
# sum_wav = np.sum(np.abs(wav))
# sum_wav_recon = np.sum(np.abs(wav_recon))
# print(sum_wav, sum_wav_recon, (sum_wav - sum_wav_recon) / sum_wav * 100)

# Write
librosa.output.write_wav('wav_orig.wav', wav, sr)
librosa.output.write_wav('wav_recon_{}.wav'.format(num_iters), wav_recon, sr)