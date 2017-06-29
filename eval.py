# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model, load_state
import os
from data import Data
from preprocess import *
from config import *


def eval():
    # Input
    x_mixed = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE), name='x_mixed')

    # Model
    net = Model(x_mixed)

    with tf.Session() as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        load_state(sess)

        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)

        data = Data(EVAL_DATA_PATH)
        wavfiles = data.next_batch(NUM_EVAL)

        # TODO refactoring
        mixed_wav = get_mixed_wav(wavfiles)
        mixed_spec = to_spectogram(mixed_wav)
        mixed = mixed_spec.transpose(0, 2, 1)

        pred = sess.run(net(), feed_dict={x_mixed: mixed})

        pred_src1, pred_src2 = pred
        pred_src1, pred_src2 = pred_src1.transpose(0, 2, 1), pred_src2.transpose(0, 2, 1)
        pred_src1_wav, pred_src2_wav = to_wav(pred_src1), to_wav(pred_src2)
        # TODO refactoring
        write_wav(mixed_wav, map(lambda f: '{}/{}'.format(EVAL_RESULT_PATH, f.replace('/', '-')), wavfiles))
        write_wav(pred_src1_wav, map(lambda f: '{}/src1-{}'.format(EVAL_RESULT_PATH, f.replace('/', '-')), wavfiles))
        write_wav(pred_src2_wav,  map(lambda f: '{}/src2-{}'.format(EVAL_RESULT_PATH, f.replace('/', '-')), wavfiles))

        writer.close()


def setup_path():
    if not os.path.exists(EVAL_RESULT_PATH):
        os.makedirs(EVAL_RESULT_PATH)


if __name__ == '__main__':
    setup_path()
    eval()