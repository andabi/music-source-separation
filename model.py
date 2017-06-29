# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from config import *
import os
from utils import shape


class Model:
    def __init__(self, n_rnn_layer=3, hidden_size=256):

        # Input, Output
        self.x_mixed = tf.placeholder(tf.float32, shape=(None, None, L_FRAME / 2 + 1), name='x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape=(None, None, L_FRAME / 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(None, None, L_FRAME / 2 + 1), name='y_src2')

        # Network
        self.hidden_size = hidden_size
        self.n_layer = n_rnn_layer
        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        rnn_layer = MultiRNNCell([GRUCell(self.hidden_size) for _ in range(self.n_layer)])
        output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x_mixed, dtype=tf.float32)
        input_size = shape(self.x_mixed)[2]
        y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src2')
        return y_hat_src1, y_hat_src2

    def loss(self):
        y_hat_src1, y_hat_src2 = self()
        return tf.reduce_mean(tf.square(self.y_src1 - y_hat_src1) + tf.square(self.y_src2 - y_hat_src2), name='loss')


def load_state(sess):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)