# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import cPickle as pickle
import sys
from pydub import AudioSegment
import os
import glob
import librosa
import numpy as np
from utils import pretty_dict

META_PATH = '/avin.hero/mss/dataset/kpop/meta.cPickle'
ARTIST_ATTR = 'artist_name_clean_exact'
SONG_ATTR = 'song_name_clean_exact'

SONG_NAME = '마지막인사'
ARTIST_NAME = ''
SOURCE_PATH = '/avin.hero/mss/dataset/kpop/mpre12k/mp3'
TARGET_PATH = '/avin.hero/mss/dataset/train/kpop'
WAV_TEMP_PATH = '/avin.hero/mss/dataset/train/melon_temp'

# Set encoding to utf-8
reload(sys)
sys.setdefaultencoding('UTF8')

# Load metadata file
meta = pickle.load(open(META_PATH, "rb"))


########################################################################################################################
# Show meta
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

# Query Inst. songs
inst_songs = query_by_inst(meta)
# print(len(songs))
# print('\n'.join('{} : {}'.format(key, value) for key, value in songs.items()))

# Search corresponding original song from inst. song
pair = dict()
for k_inst, v_inst in inst_songs.items():
    song_name = v_inst[SONG_ATTR]
    artist_name = v_inst[ARTIST_ATTR][0]
    song_length = v_inst['song_length']
    original = [k for k, v in meta.iteritems() if by_song_artist(v, song_name, artist_name)
                and by_inst(v, False)
                and by_song_length(v, song_length)]
    pair[k_inst] = original
# TODO encoding
# print(pretty_dict(pair))


########################################################################################################################
# Convert mp3 to wav
########################################################################################################################

# Necessary libraries: ffmpeg, libav
def convert_mp3_to_wav(source_path, target_path):
    basepath, filename = os.path.split(source_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(source_path).export(target_path, format='wav')

sources = []
original_song_list = sum(pair.itervalues(), [])
inst_song_list = list(inst_songs.itervalues())
sources.extend(original_song_list)
sources.extend(inst_songs)
for s in sources:
    search_path = '{}/*/*/{}.mp3'.format(SOURCE_PATH, s)
    for source_path in glob.glob(search_path):
        target_path = '{}/{}.wav'.format(WAV_TEMP_PATH, s)
        convert_mp3_to_wav(source_path, target_path)


########################################################################################################################
# Create 2-channel wav
########################################################################################################################

for inst, originals in pair.iteritems():
    if not originals:
        continue
    else:
        orig = originals[0]
        inst_path = '{}/{}.wav'.format(WAV_TEMP_PATH, inst)
        orig_path = '{}/{}.wav'.format(WAV_TEMP_PATH, orig)
        inst_data, sr = librosa.load(inst_path, mono=True, duration=60)
        orig_data, sr = librosa.load(orig_path, mono=True, duration=60)

        assert(len(inst_data) == len(orig_data))

        vocal_data = orig_data - inst_data
        mixed = np.array([inst_data, vocal_data])
        song_name = inst_songs[inst][SONG_ATTR]
        mixed_path = '{}/{}.wav'.format(TARGET_PATH, song_name)
        librosa.output.write_wav(mixed_path, mixed, sr)