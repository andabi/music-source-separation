# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Write
CASE = 'enhanced_loss'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE

# Model
BATCH_SIZE = 8
SR = 16000
L_FRAME = 4096
L_HOP = L_FRAME / 2

# Train
TRAIN_DATA_PATH = 'dataset/ikala'
LR = 0.0001
FINAL_STEP = 10000
CKPT_STEP = 100
RE_TRAIN = False

# Eval
EVAL_DATA_PATH = 'dataset/MIR-1K/Wavfile'
EVAL_RESULT_PATH = 'results/' + CASE
NUM_EVAL = 1