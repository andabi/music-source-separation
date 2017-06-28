# -*- coding: utf-8 -*-
# !/usr/bin/env python

import random
from os import walk


class Data:
    def __init__(self, path):
        self.path = path

    def next_batch(self, size=10):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, size)
        return wavfiles

# train_data = Data('dataset/ikala')
