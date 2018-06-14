# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi

Edited to include unet structure by Joel Lof. joel.lof@icloud.com
'''

# from keras import backend as keras
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from __future__ import division

import os

from config import ModelConfig

from keras.layers import Concatenate, Conv2D, Dropout, Input, MaxPooling2D
# merge, Cropping2D
from keras.layers import UpSampling2D
from keras.models import *
from keras.optimizers import *

import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell

from utils import shape

# for unet


class Model:
    '''[summary]

    [description]
    '''
    def __init__(self, n_rnn_layer=3, hidden_size=256):
        '''[summary]

        [description]

        Keyword Arguments:
            n_rnn_layer {number} -- [description] (default: {3})
            hidden_size {number} -- [description] (default: {256})
        '''

        # Input, Output
        # Tensor("x_mixed:0", shape=(?, ?, 513), dtype=float32)
        self.x_mixed = tf.placeholder(tf.float32, shape=(
            None, None, ModelConfig.L_FRAME // 2 + 1), name='x_mixed')
        self.x_mixed_unet = tf.placeholder(tf.float32, shape=(
            None, None, 1, ModelConfig.L_FRAME // 2 + 1), name='x_mixed_unet')
        self.y_src1 = tf.placeholder(tf.float32, shape=(
            None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(
            None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src2')

        # Network
        self.hidden_size = hidden_size  # 256
        self.n_layer = n_rnn_layer      # 3
        # name='net', func =_net ''returns A function to encapsulate a set of
        # variables which should be created once and reused.''
        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def unet(self, input_size):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal'
                     )(UpSampling2D(size=(2, 2))(drop5))
        merge6 = Concatenate(axis=3)([drop4, up6])  # usr add
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal'
                     )(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])  # usr add
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal'
                     )(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])  # usr add
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal'
                     )(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])  # usr add
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        return conv10

    def _net(self):
        # RNN and dense layers
        # 256 _ in 3
        # returns Input tensor or list of input tensors. #class
        # MultiRNNCell(RNNCell):
        rnn_layer = MultiRNNCell([GRUCell(self.hidden_size)
                                  for _ in range(self.n_layer)])
        output_rnn, rnn_state = tf.nn.dynamic_rnn(
            rnn_layer, self.x_mixed, dtype=tf.float32)
        input_size = shape(self.x_mixed)[2]

        y_hat_src1 = tf.layers.dense(
            inputs=output_rnn, units=input_size, activation=tf.nn.relu,
            name='y_hat_src1')
        # y_hat_src1 = Conv2D(513, 3, activation = 'relu',
        # padding = 'same', kernel_initializer = 'he_normal')
        # (self.x_mixed_unet)

        y_hat_src2 = tf.layers.dense(
            inputs=output_rnn, units=input_size, activation=tf.nn.relu,
            name='y_hat_src2')

        # input_size = shape(self.x_mixed)[2]
        # y_hat_src1 = unet(input_size = input_size)
        # y_hat_src2 = unet(input_size = input_size)
        # model = Model(inputs = inputs, outputs = conv10)
        # model.compile(optimizer = Adam(lr = 1e-4),
        # loss = 'binary_crossentropy', metrics = ['accuracy'])

        # time-freq masking layer
        y_tilde_src1 = y_hat_src1 / \
            (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / \
            (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed

        return y_tilde_src1, y_tilde_src2

    def loss(self):
        pred_y_src1, pred_y_src2 = self()
        return tf.reduce_mean(tf.square(self.y_src1 - pred_y_src1) +
                              tf.square(self.y_src2 - pred_y_src2),
                              name='loss')

    @staticmethod
    # shape = (batch_size, n_freq, n_frames) => (batch_size, n_frames, n_freq)
    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % ModelConfig.SEQ_LEN > 0:
            pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width,
                            mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1),
                           (-1, ModelConfig.SEQ_LEN, freq))
        return batch, padded_src

    @staticmethod
    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq,
        # n_frames)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(
            os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
