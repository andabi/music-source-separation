# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model, load_state
import os
from data import Data
from preprocess import *
from config import EvalConfig
from mir_eval.separation import bss_eval_sources
import shutil


def eval():
    # Model
    model = Model()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.Session(config=EvalConfig.session_conf) as sess:
        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        load_state(sess, EvalConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(EvalConfig.GRAPH_PATH, sess.graph)

        data = Data(EvalConfig.DATA_PATH)
        mixed_wav, src1_wav, src2_wav = data.next_wavs(EvalConfig.NUM_EVAL, sec=EvalConfig.SECONDS)

        # TODO refactoring
        mixed_spec = to_spectrogram(mixed_wav)
        mixed_mag = get_magnitude(mixed_spec)
        mixed_phase = get_phase(mixed_spec)
        mixed_batched = model.seq_to_batch(mixed_mag)

        # assert(np.equal(mixed_spec, get_stft_matrix(mixed_magnitude, mixed_phase)).all())
        # a = np.around(mixed_spec, 3)
        # b = np.around(get_stft_matrix(mixed_magnitude, mixed_phase), 3)
        # print((a == b).all())

        pred = sess.run(model(), feed_dict={model.x_mixed: mixed_batched})

        # (magnitude, phase) -> spectrogram -> wav
        pred_src1_mag, pred_src2_mag = pred
        pred_src1_mag = model.batch_to_seq(pred_src1_mag, EvalConfig.NUM_EVAL)
        pred_src2_mag = model.batch_to_seq(pred_src2_mag, EvalConfig.NUM_EVAL)
        mixed_phase = mixed_phase[:, :, :pred_src1_mag.shape[-1]]
        pred_src1_spec = get_stft_matrix(pred_src1_mag, mixed_phase)
        pred_src2_spec = get_stft_matrix(pred_src2_mag, mixed_phase)
        pred_src1_wav, pred_src2_wav = to_wav(pred_src1_spec), to_wav(pred_src2_spec)

        # # (magnitude, phase) -> spectrogram -> wav (with smoothing using t-f mask)
        mask_src1 = time_freq_mask(pred_src1_mag, pred_src2_mag)
        mask_src2 = 1.0 - mask_src1
        smoothed_pred_src1_magnitude = pred_src1_mag * mask_src1
        smoothed_pred_src2_magnitude = pred_src2_mag * mask_src2
        smoothed_pred_src1_spec, smoothed_pred_src2_spec = get_stft_matrix(smoothed_pred_src1_magnitude, mixed_phase), \
                                                           get_stft_matrix(smoothed_pred_src2_magnitude, mixed_phase)
        smoothed_pred_src1_wav, smoothed_pred_src2_wav = to_wav(smoothed_pred_src1_spec), to_wav(
            smoothed_pred_src2_spec)

        # print(np.max((mixed_wav[:,:60000] - pred_src2_wav[:,:60000] - pred_src1_wav[:,:60000])))

        # TODO refactoring
        tf.summary.audio('GT_mixed', mixed_wav, ModelConfig.SR)
        # tf.summary.audio('GT_music', src1_wav, ModelConfig.SR)
        # tf.summary.audio('GT_vocal', src2_wav, ModelConfig.SR)
        tf.summary.audio('P_mixed', pred_src1_wav + pred_src2_wav, ModelConfig.SR)
        # tf.summary.audio('P_mixed_smoothed', smoothed_pred_src1_wav + smoothed_pred_src2_wav, ModelConfig.SR)
        tf.summary.audio('P_music', pred_src1_wav, ModelConfig.SR)
        # tf.summary.audio('P_music_smoothed', smoothed_pred_src1_wav, ModelConfig.SR)
        tf.summary.audio('P_vocal', pred_src2_wav, ModelConfig.SR)
        # tf.summary.audio('P_vocal_smoothed', smoothed_pred_src2_wav, ModelConfig.SR)

        # TODO refactoring
        # BSS metrics
        crop_len_src1 = min(pred_src1_wav.shape[-1], src1_wav.shape[-1])
        crop_len_src2 = min(pred_src2_wav.shape[-1], src2_wav.shape[-1])
        pred_src1_wav = pred_src1_wav[0][:crop_len_src1]
        src1_wav = src1_wav[0][:crop_len_src1]
        pred_src2_wav = pred_src2_wav[0][:crop_len_src2]
        src2_wav = src2_wav[0][:crop_len_src2]
        # crop_len_src1 = min(smoothed_pred_src1_wav.shape[-1], src1_wav.shape[-1])
        # crop_len_src2 = min(smoothed_pred_src2_wav.shape[-1], src2_wav.shape[-1])
        # smoothed_pred_src1_wav = smoothed_pred_src1_wav[0][:crop_len_src1]
        # src1_wav = src1_wav[0][:crop_len_src1]
        # smoothed_pred_src2_wav = smoothed_pred_src2_wav[0][:crop_len_src2]
        # src2_wav = src2_wav[0][:crop_len_src2]
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                            np.array([pred_src1_wav, pred_src2_wav]), False)
        tf.summary.scalar('music_sdr', sdr[0])
        tf.summary.scalar('music_sir', sir[0])
        tf.summary.scalar('music_sar', sar[0])
        tf.summary.scalar('vocal_sdr', sdr[1])
        tf.summary.scalar('vocal_sir', sir[1])
        tf.summary.scalar('vocal_sar', sar[1])

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=global_step.eval())

        writer.close()


def setup_path():
    if EvalConfig.RE_EVAL:
        if os.path.exists(EvalConfig.GRAPH_PATH):
            shutil.rmtree(EvalConfig.GRAPH_PATH)

    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)


if __name__ == '__main__':
    setup_path()
    eval()
