# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Write
CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE

# Model
INPUT_SIZE = 2049
BATCH_SIZE = 4

# Train
TRAIN_DATA_PATH = 'dataset/ikala'
LR = 0.00001
FINAL_STEP = 1000
CKPT_STEP = 10
RE_TRAIN = False

# Eval
EVAL_DATA_PATH = 'dataset/ikala'
EVAL_RESULT_PATH = 'results/' + CASE
NUM_EVAL = 3