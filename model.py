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

        # network
        self.net = tf.make_template('net', self._net)

    def __call__(self, input_seq):
        return self.net(input_seq)

    def _net(self, input_seq):
        self.input_seq = input_seq
        self.rnn_layer = MultiRNNCell([GRUCell(self.hidden_size, self.input_size) for _ in range(self.n_layer)])
        output_rnn, rnn_state = tf.nn.dynamic_rnn(self.rnn_layer, self.input_seq, dtype=tf.float32)
        self.y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=self.input_size, activation=tf.nn.relu, name='y_hat_src1')
        self.y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=self.input_size, activation=tf.nn.relu, name='y_hat_src2')
        return self.y_hat_src1, self.y_hat_src2

    def loss(self, y_src1, y_src2):
        return tf.reduce_mean(tf.square(y_src1 - self.y_hat_src1) + tf.square(y_src2 - self.y_hat_src2), name='loss')