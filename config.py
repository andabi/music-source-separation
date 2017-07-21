# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf


# Model
class ModelConfig:
    SR = 16000
    L_FRAME = 1024
    L_HOP = L_FRAME / 4
    SEQ_LEN = 1


# Train
class TrainConfig:
    CASE = 'ikala_kpop'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = 'dataset/train'
    LR = 0.0001
    FINAL_STEP = 100000
    CKPT_STEP = 500
    NUM_WAVFILE = 1
    SECONDS = 30
    RE_TRAIN = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
    )


# Eval
class EvalConfig:
    CASE = 'ikala_kpop'
    # CASE = '4-frames-masking-layer'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/eval/kpop'
    # DATA_PATH = 'dataset/mir-1k/Wavfile'
    # DATA_PATH = 'dataset/ikala'
    MAG_ONLY = True
    NUM_EVAL = 2
    SECONDS = 15
    RE_EVAL = True
    EVAL_METRIC = False
    WRITE_RESULT = False
    RESULT_PATH = 'results/' + CASE
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True)
    )