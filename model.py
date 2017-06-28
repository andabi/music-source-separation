# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell


class Model:
    def __init__(self, input_size, batch_size=32, n_rnn_layer=3, hidden_size=256):
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layer = n_rnn_layer
        self.net = tf.make_template('net', self._net)

    def __call__(self, x_mixed):
        return self.net(x_mixed)

    def _net(self, x_mixed):
        rnn_layer = MultiRNNCell([GRUCell(self.hidden_size) for _ in range(self.n_layer)])
        output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, x_mixed, dtype=tf.float32)
        y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=self.input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=self.input_size, activation=tf.nn.relu, name='y_hat_src2')
        return y_hat_src1, y_hat_src2

    def loss(self, x_mixed, y_src1, y_src2):
        y_hat_src1, y_hat_src2 = self.net(x_mixed)
        return tf.reduce_mean(tf.square(y_src1 - y_hat_src1) + tf.square(y_src2 - y_hat_src2), name='loss')