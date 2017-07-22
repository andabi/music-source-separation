# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import audio_utils
import cPickle as pickle
import os
import glob
import librosa
import numpy as np
from utils import pretty_dict
import pprint

META_PATH = '/avin.hero/mss/dataset/kpop/meta.cPickle'
ARTIST_ATTR = 'artist_name_clean_exact'
SONG_ATTR = 'song_name_clean_exact'

# Load metadata file
meta = pickle.load(open(META_PATH, "rb"))  # EUC-KR

########################################################################################################################
# Display meta
########################################################################################################################

# Show artist list
# artists = sorted(set(sum([v[ARTIST_ATTR] for k, v in meta.iteritems()], [])))
# target_artist = ', '.join([a for a in artists if a == ARTIST_NAME])
# n_artists = len(artists)
# print('\n'.join(artists))
# print(target_artist)


########################################################################################################################
# Query songs by condition
########################################################################################################################

SONG_NAME = ''
ARTIST_NAME = '볼빨간사춘기'


def query_by_song_artist(meta, song_name=None, artist_name=None):
    return dict((k, v) for k, v in meta.iteritems() if by_song_artist(v, song_name, artist_name))


def query_by_inst(meta, is_inst=True):
    return dict((k, v) for k, v in meta.iteritems() if by_inst(v, is_inst))


def by_song_length(value_dict, len):
    return value_dict['song_length'] == len


def by_song_artist(value_dict, song_name=None, artist_name=None):
    filter_song = song_name == value_dict[SONG_ATTR] or not song_name
    filter_artist = len([a for a in value_dict[ARTIST_ATTR] if artist_name == a]) > 0 or not artist_name
    return filter_song and filter_artist


def by_inst(value_dict, is_inst=True):
    if is_inst: flag = 'Y'
    else:  flag = 'N'
    return value_dict['mr_yn'] == flag

# Query songs
songs = query_by_song_artist(meta, SONG_NAME, ARTIST_NAME)
song_title_dict = dict(map(lambda (k, v): (k, v[SONG_ATTR]), songs.items()))
print(pretty_dict(song_title_dict))

# Query Inst. songs
inst_songs = query_by_inst(meta)
# print(len(songs))
# print(pretty_dict(songs))


########################################################################################################################
# Search corresponding original song from inst. song
########################################################################################################################

# pair = dict()
# for k_inst, v_inst in inst_songs.items():
#     song_name = v_inst[SONG_ATTR]
#     artist_name = v_inst[ARTIST_ATTR][0]
#     song_length = v_inst['song_length']
#     original = [k for k, v in meta.iteritems() if by_song_artist(v, song_name, artist_name)
#                 and by_inst(v, False)
#                 and by_song_length(v, song_length)]
#     pair[k_inst] = original
# print(pretty_dict(pair))


########################################################################################################################
# Write songs
########################################################################################################################

SOURCE_PATH = '/avin.hero/mss/dataset/kpop/mpre12k/mp3'
TARGET_PATH = '/avin.hero/mss/dataset/kpop'

# Write mp3 to wav
for s in song_title_dict:
    search_path = '{}/*/*/{}.mp3'.format(SOURCE_PATH, s)
    for source_path in glob.glob(search_path):
        target_path = '{}/{}.wav'.format(TARGET_PATH, song_title_dict[s])
        audio_utils.rewrite_mp3_to_wav(source_path, target_path)

# Search and rewrite mp3 to wav

# songs = []
# original_song_list = sum(pair.itervalues(), [])
# inst_song_list = list(inst_songs.itervalues())
# songs.extend(original_song_list)
# songs.extend(inst_songs)
# for s in songs:
#     search_path = '{}/*/*/{}.mp3'.format(SOURCE_PATH, s)
#     for source_path in glob.glob(search_path):
#         target_path = '{}/{}.wav'.format(TARGET_PATH, s)
#         audio_utils.rewrite_mp3_to_wav(source_path, target_path)


########################################################################################################################
# Create 2-channel wav
########################################################################################################################

# SOURCE_PATH = '/avin.hero/mss/dataset/kpop'
# TARGET_PATH = '/avin.hero/mss/dataset/train/kpop'


# for inst, originals in pair.iteritems():
#     if not originals:
#         continue
#     else:
#         orig = originals[0]
#         inst_path = '{}/{}.wav'.format(SOURCE_PATH, inst)
#         orig_path = '{}/{}.wav'.format(SOURCE_PATH, orig)
#         inst_data, sr = librosa.load(inst_path, mono=True, duration=60)
#         orig_data, sr = librosa.load(orig_path, mono=True, duration=60)
#
#         assert(len(inst_data) == len(orig_data))
#
#         vocal_data = orig_data - inst_data
#         mixed = np.array([inst_data, vocal_data])
#         song_name = inst_songs[inst][SONG_ATTR]
#         mixed_path = '{}/{}.wav'.format(TARGET_PATH, song_name)
#         librosa.output.write_wav(mixed_path, mixed, sr)
