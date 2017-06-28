# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model
import os
import shutil
from data import Data
from preprocess import *
from utils import Diff

CASE = 'default'
DATA_PATH = 'dataset/ikala'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE
RE_TRAIN = False
INPUT_SIZE = 2049
BATCH_SIZE = 2
LR = 0.0001
FINAL_STEP = 1000
CKPT_STEP = 10


def train():
    net = Model(INPUT_SIZE)
    input = Data(DATA_PATH)
    x_mixed = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, INPUT_SIZE), name='x_mixed')
    y_src1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, INPUT_SIZE), name='y_src1')
    y_src2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, INPUT_SIZE), name='y_src2')

    loss_op = net.loss(x_mixed, y_src1, y_src2)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_op, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_state(sess)

        loss = Diff()
        for step in range(global_step.eval(), FINAL_STEP):
            wavfiles = input.next_batch(BATCH_SIZE)

            mixed_wav = get_mixed_wav(wavfiles)
            mixed_spec = to_spectogram(mixed_wav)
            mixed = mixed_spec.transpose(0, 2, 1)

            src1_wav, src2_wav = get_src1_src2_wav(wavfiles)
            src1_spec, src2_spec = to_spectogram(src1_wav), to_spectogram(src2_wav)
            src1, src2 = src1_spec.transpose(0, 2, 1), src2_spec.transpose(0, 2, 1)

            l, _ = sess.run([loss_op, optimizer], feed_dict={x_mixed: mixed, y_src1: src1, y_src2: src2})
            loss.update(l)
            print('step-{}\td_loss={}\tloss={}'.format(step, loss.diff * 100, loss.value))

            # save state
            if step % CKPT_STEP == 0:
                tf.train.Saver().save(sess, CKPT_PATH + '/checkpoint', global_step=step)

        # Write graph
        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)
        writer.close()


def load_state(sess):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)


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
