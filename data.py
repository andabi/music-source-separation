# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

class Data:
    def __init__(self, path):
        self.path = path
        self.index = 0
        self.wavfiles = []

    def load_data(self):
        for (root, dirs, files) in walk(self.path):  # go through the dataset
            self.wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")]) #read all filename into wavfiles

    def next_batch(self, sec):
        n_begin = self.index
        n_end = n_begin + TrainConfig.BATCH_SIZE
        if n_end >= TrainConfig.NUM_WAVFILE:
            self.index = 0
            n_begin = self.index
            n_end = n_begin + TrainConfig.BATCH_SIZE
        self.index += TrainConfig.BATCH_SIZE
        current_wavs = self.wavfiles[n_begin:n_end]
        mixed, src1, src2 = get_random_wav(current_wavs, sec, ModelConfig.SR)
        print('index in batch:{}-{}'.format(n_begin,n_end))
        return mixed, src1, src2, current_wavs
    
    def total_batches(self):
        return TrainConfig.NUM_WAVFILE // TrainConfig.BATCH_SIZE

    def get_wavs(self, sec, size=1):
        #wavfiles = []
        #for (root, dirs, files) in walk(self.path):  # go through the dataset
        #    wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")]) #read all filename into wavfiles
        wavfiles = random.sample(self.wavfiles, size)
        print('wavfiles:{}'.format(wavfiles))
        mixed, src1, src2 = get_random_wav(wavfiles, sec, ModelConfig.SR)
        return mixed, src1, src2, wavfiles
