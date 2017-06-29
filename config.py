# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Write
CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE

# Model
BATCH_SIZE = 16
SR = 16000
L_FRAME = 4096
L_HOP = L_FRAME / 2

# Train
TRAIN_DATA_PATH = 'dataset/ikala'
LR = 0.005
FINAL_STEP = 1000
CKPT_STEP = 10
RE_TRAIN = True

# Eval
EVAL_DATA_PATH = 'dataset/ikala'
EVAL_RESULT_PATH = 'results/' + CASE
NUM_EVAL = 1