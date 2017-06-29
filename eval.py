# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model, load_state
import os
from data import Data
from preprocess import *
from config import *


def eval():
    # Model
    model = Model()

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

        pred = sess.run(model(), feed_dict={model.x_mixed: mixed})

        pred_src1, pred_src2 = pred
        pred_src1, pred_src2 = pred_src1.transpose(0, 2, 1), pred_src2.transpose(0, 2, 1)
        pred_src1_wav, pred_src2_wav = to_wav(pred_src1), to_wav(pred_src2)
        # TODO refactoring
        tf.summary.audio('mixed', mixed_wav, SR)
        tf.summary.audio('pred_src1', pred_src1_wav, SR)
        tf.summary.audio('pred_src2', pred_src2_wav, SR)
        tf.summary.audio('pred_mixed', pred_src1_wav + pred_src2_wav, SR)
        writer.add_summary(sess.run(tf.summary.merge_all()))

        writer.close()


def setup_path():
    if not os.path.exists(EVAL_RESULT_PATH):
        os.makedirs(EVAL_RESULT_PATH)


if __name__ == '__main__':
    setup_path()
    eval()