# -*- coding: utf-8 -*-
# !/usr/bin/env python

import random
from os import walk
from config import ModelConfig
from preprocess import get_mixed_wav, get_src1_src2_wav


class Data:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, size=1):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, size)
        mixed = get_mixed_wav(wavfiles, ModelConfig.SR)
        src1, src2 = get_src1_src2_wav(wavfiles, ModelConfig.SR)
        return mixed, src1, src2