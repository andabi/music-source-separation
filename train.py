# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
from model import Model
import os
import shutil

CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE
RE_TRAIN = False
INPUT_SIZE = 8000
BATCH_SIZE = 32
SEQ_LEN = 30
LR = 0.005


def train():
    net = Model(INPUT_SIZE)

    x_mixed = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQ_LEN, INPUT_SIZE), name='x')
    y_src1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQ_LEN, INPUT_SIZE), name='y_src1')
    y_src2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQ_LEN, INPUT_SIZE), name='y_src2')
    y_hat_src1, y_hat_src2 = net(x_mixed)

    loss = net.loss(y_src1, y_src2)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # out, _ = sess.run(out, optimizer, feed_dict={})

        # Write graph
        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)
        writer.close()


def clear_dir():
    if RE_TRAIN:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
        if os.path.exists(GRAPH_PATH):
            shutil.rmtree(GRAPH_PATH)
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)


if __name__ == '__main__':
    # TODO multi-gpu
    # TODO queue
    clear_dir()
    train()