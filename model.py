# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import division
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from config import ModelConfig
import os
from utils import shape
import numpy as np


class Model:
    def __init__(self, n_rnn_layer=3, hidden_size=256):

        # Input, Output
        self.x_mixed = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src2')

        # Network
        self.hidden_size = hidden_size
        self.n_layer = n_rnn_layer
        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        # RNN and dense layers
        rnn_layer = MultiRNNCell([GRUCell(self.hidden_size) for _ in range(self.n_layer)])
        output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x_mixed, dtype=tf.float32)
        input_size = shape(self.x_mixed)[2]
        y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src2')

        # time-freq masking layer
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed

        return y_tilde_src1, y_tilde_src2

    def loss(self):
        pred_y_src1, pred_y_src2 = self()
        return tf.reduce_mean(tf.square(self.y_src1 - pred_y_src1) + tf.square(self.y_src2 - pred_y_src2), name='loss')


# TODO batch padding (don't crop tail)
# shape = (B, F, T) => (B, T, F)
def seq_to_batch(src):
    src = src.transpose(0, 2, 1)
    num_wavs, n_frame, freq = src.shape
    seq_len = num_wavs * n_frame // ModelConfig.BATCH_SIZE + 1

    # crop tail frames which is smaller than seq_len
    n_frame -= (n_frame % seq_len)
    src = src[:, :n_frame]

    src = np.array(np.hsplit(src, n_frame // seq_len))
    src = np.reshape(src, (-1, seq_len, freq))
    return src


# shape = (B, T, F) => (B, F, T)
def batch_to_seq(pred_src, num_file):
    batch_size, n_frame, freq = pred_src.shape
    pred_src = np.reshape(pred_src, (num_file, -1, freq))
    pred_src = pred_src.transpose(0, 2, 1)
    return pred_src


def load_state(sess, ckpt_path):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)