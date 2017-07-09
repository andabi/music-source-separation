# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf


# Model
class ModelConfig:
    BATCH_SIZE = 64
    SR = 16000
    L_FRAME = 1024
    L_HOP = L_FRAME / 4


# Train
class TrainConfig:
    CASE = 'ikala-mir1k'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = 'dataset/train'
    LR = 0.0005
    FINAL_STEP = 50000
    CKPT_STEP = 100
    RE_TRAIN = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
    )


# Eval
class EvalConfig:
    CASE = 'ikala-mir1k'
    # CASE = '4-frames'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/eval'
    # DATA_PATH = 'dataset/mir-1k/Wavfile'
    # DATA_PATH = 'dataset/ikala'
    RESULT_PATH = 'results/' + CASE
    NUM_EVAL = 1
    SECONDS = 30
    RE_EVAL = True
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True)
    )