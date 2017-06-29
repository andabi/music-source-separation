# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model, load_state
import os
import shutil
from data import Data
from preprocess import *
from utils import Diff
from config import *


def train():
    # Model
    model = Model()

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_fn, global_step=global_step)

    # Summaries
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss_fn, v))
    tf.summary.scalar('loss', loss_fn)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        load_state(sess)

        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)

        data = Data(TRAIN_DATA_PATH)
        loss = Diff()
        for step in range(global_step.eval(), FINAL_STEP):
            wavfiles = data.next_batch(BATCH_SIZE)

            mixed_wav = get_mixed_wav(wavfiles)
            mixed_spec = to_spectogram(mixed_wav)
            mixed = mixed_spec.transpose(0, 2, 1)

            src1_wav, src2_wav = get_src1_src2_wav(wavfiles)
            src1_spec, src2_spec = to_spectogram(src1_wav), to_spectogram(src2_wav)
            src1, src2 = src1_spec.transpose(0, 2, 1), src2_spec.transpose(0, 2, 1)

            l, _, summary = sess.run([loss_fn, optimizer, summary_op],
                                     feed_dict={model.x_mixed: mixed, model.y_src1: src1, model.y_src2: src2})

            loss.update(l)
            print('step-{}\td_loss={:2.2f}\tloss={}'.format(step, loss.diff * 100, loss.value))

            # Save state
            if step % CKPT_STEP == 0:
                tf.train.Saver().save(sess, CKPT_PATH + '/checkpoint', global_step=step)
                writer.add_summary(summary, global_step=step)

        writer.close()


def setup_path():
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
    setup_path()
    train()
