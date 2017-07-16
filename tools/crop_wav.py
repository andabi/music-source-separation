# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import librosa

SECONDS = 30
FILE_PATH = '../dataset/kpop/Lucifer.wav'

data, sr = librosa.load(FILE_PATH, mono=False, duration=SECONDS)
librosa.output.write_wav(FILE_PATH.replace('.wav', '_' + str(SECONDS) + 's.wav'), data, sr)