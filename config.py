# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf


# Model
class ModelConfig:
    SR = 32000
    L_FRAME = 2048
    L_HOP = L_FRAME / 4
    SEQ_LEN = 4


# Train
class TrainConfig:
    CASE = '4frame-sr32000-nframe2048'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = 'dataset/ikala'
    LR = 0.00005
    FINAL_STEP = 100000
    CKPT_STEP = 100
    SECONDS = 30
    RE_TRAIN = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
    )


# Eval
class EvalConfig:
    CASE = '4frame-sr32000-nframe2048'
    # CASE = '4-frames-masking-layer'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/eval/kpop'
    # DATA_PATH = 'dataset/mir-1k/Wavfile'
    # DATA_PATH = 'dataset/ikala'
    RESULT_PATH = 'results/' + CASE
    NUM_EVAL = 2
    SECONDS = 30
    RE_EVAL = True
    EVAL_METRIC = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True)
    )