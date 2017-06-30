# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from model import Model, load_state
import os
from data import Data
from preprocess import *
from config import *
from mir_eval.separation import bss_eval_sources

def eval():
    # Model
    model = Model()

    with tf.Session() as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        load_state(sess)

        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)

        data = Data(EVAL_DATA_PATH)
        wavfiles = data.next_batch(NUM_EVAL)

        # TODO refactoring
        mixed_wav = get_mixed_wav(wavfiles)
        mixed_spec = to_spectrogram(mixed_wav)
        mixed_magnitude = get_magnitude(mixed_spec)
        mixed_phase = get_phase(mixed_spec)
        mixed = mixed_magnitude.transpose(0, 2, 1)

        # assert(np.equal(mixed_spec, get_stft_matrix(mixed_magnitude, mixed_phase)).all())
        # a = np.around(mixed_spec, 3)
        # b = np.around(get_stft_matrix(mixed_magnitude, mixed_phase), 3)
        # print((a == b).all())

        src1_wav, src2_wav = get_src1_src2_wav(wavfiles)

        pred = sess.run(model(), feed_dict={model.x_mixed: mixed})

        # (magnitude, phase) -> spectrogram -> wav
        pred_src1_magnitude, pred_src2_magnitude = pred
        pred_src1_magnitude, pred_src2_magnitude = pred_src1_magnitude.transpose(0, 2, 1), \
                                                   pred_src2_magnitude.transpose(0, 2, 1)
        pred_src1_spec, pred_src2_spec = get_stft_matrix(pred_src1_magnitude, mixed_phase), get_stft_matrix(
            pred_src2_magnitude, mixed_phase)
        pred_src1_wav, pred_src2_wav = to_wav(pred_src1_spec), to_wav(pred_src2_spec)

        # # (magnitude, phase) -> spectrogram -> wav (with smoothing using t-f mask)
        mask_src1 = time_freq_mask(pred_src1_magnitude, pred_src2_magnitude)
        mask_src2 = 1.0 - mask_src1
        smoothed_pred_src1_magnitude = pred_src1_magnitude * mask_src1
        smoothed_pred_src2_magnitude = pred_src2_magnitude * mask_src2
        smoothed_pred_src1_spec, smoothed_pred_src2_spec = get_stft_matrix(smoothed_pred_src1_magnitude, mixed_phase), \
                                                           get_stft_matrix(smoothed_pred_src2_magnitude, mixed_phase)
        smoothed_pred_src1_wav, smoothed_pred_src2_wav = to_wav(smoothed_pred_src1_spec), to_wav(smoothed_pred_src2_spec)

        # print(np.max((mixed_wav[:,:60000] - pred_src2_wav[:,:60000] - pred_src1_wav[:,:60000])))

        # TODO refactoring
        tf.summary.audio('mixed', mixed_wav, SR)
        tf.summary.audio('pred_music', pred_src1_wav, SR)
        tf.summary.audio('pred_music_smoothed', smoothed_pred_src1_wav, SR)
        tf.summary.audio('pred_vocal', pred_src2_wav, SR)
        tf.summary.audio('pred_vocal_smoothed', smoothed_pred_src2_wav, SR)
        tf.summary.audio('pred_mixed', pred_src1_wav + pred_src2_wav, SR)
        tf.summary.audio('pred_mixed_smoothed', smoothed_pred_src1_wav + smoothed_pred_src2_wav, SR)
        # tf.summary.audio('reverse_mixed', np.flip(mixed_wav, -1), SR)
        # tf.summary.audio('reverse_src1', np.flip(src1_wav, -1), SR)
        # tf.summary.audio('reverse_src2', np.flip(src2_wav, -1), SR)

        # TODO refactoring
        # BSS metrics
        crop_len_src1 = min(smoothed_pred_src1_wav.shape[-1], src1_wav.shape[-1])
        crop_len_src2 = min(smoothed_pred_src2_wav.shape[-1], src2_wav.shape[-1])
        smoothed_pred_src1_wav = smoothed_pred_src1_wav[0][:crop_len_src1]
        src1_wav = src1_wav[0][:crop_len_src1]
        smoothed_pred_src2_wav = smoothed_pred_src2_wav[0][:crop_len_src2]
        src2_wav = src2_wav[0][:crop_len_src2]
        sdr, sir, sar, _ = bss_eval_sources(np.array([smoothed_pred_src1_wav, smoothed_pred_src2_wav]),
                                            np.array([src1_wav, src2_wav]))
        tf.summary.scalar('sdr_music', sdr[0])
        tf.summary.scalar('sir_music', sir[0])
        tf.summary.scalar('sar_music', sar[0])
        tf.summary.scalar('sdr_vocal', sdr[1])
        tf.summary.scalar('sir_vocal', sir[1])
        tf.summary.scalar('sar_vocal', sar[1])

        writer.add_summary(sess.run(tf.summary.merge_all()))

        writer.close()


def setup_path():
    if not os.path.exists(EVAL_RESULT_PATH):
        os.makedirs(EVAL_RESULT_PATH)


if __name__ == '__main__':
    setup_path()
    eval()