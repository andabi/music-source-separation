# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf
from model import Model, load_state, spec_to_batch
import os
import shutil
from data import Data
from preprocess import to_spectrogram, get_magnitude
from utils import Diff
from config import TrainConfig


def train():
    # Model
    model = Model()

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    # Summaries
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss_fn, v))
    tf.summary.scalar('loss', loss_fn)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    summary_op = tf.summary.merge_all()

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        load_state(sess, TrainConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        data = Data(TrainConfig.DATA_PATH)
        loss = Diff()
        for step in range(global_step.eval(), TrainConfig.FINAL_STEP):
            mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(TrainConfig.SECONDS, TrainConfig.NUM_WAVFILE)

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)

            src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

            src1_batch, src1_mag = spec_to_batch(src1_mag)
            src2_batch, src1_mag = spec_to_batch(src2_mag)
            mixed_batch, mixed_mag = spec_to_batch(mixed_mag)

            l, _, summary = sess.run([loss_fn, optimizer, summary_op],
                                     feed_dict={model.x_mixed: mixed_batch, model.y_src1: src1_batch, model.y_src2: src2_batch})

            loss.update(l)
            print('step-{}\td_loss={:2.2f}\tloss={}'.format(step, loss.diff * 100, loss.value))

            writer.add_summary(summary, global_step=step)

            # Save state
            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)

        writer.close()


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


if __name__ == '__main__':
    # TODO multi-gpu, queue, parellel
    setup_path()
    train()
