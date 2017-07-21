# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import random
from os import walk
from config import ModelConfig
from preprocess import get_random_wav


class Data:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, sec, size=1):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, size)
        mixed, src1, src2 = get_random_wav(wavfiles, sec, ModelConfig.SR)
        return mixed, src1, src2, wavfiles