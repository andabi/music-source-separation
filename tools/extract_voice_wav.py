# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import librosa

SECONDS = 60
MIXED_FILE_PATH = '../dataset/kpop/30414090.wav'
MUSIC_FILE_PATH = '../dataset/kpop/30416828.wav'
VOICE_FILE_PATH = '../dataset/kpop/voice.wav'

mixed, sr = librosa.load(MIXED_FILE_PATH, mono=False, duration=SECONDS)
music, _ = librosa.load(MUSIC_FILE_PATH, mono=False, duration=SECONDS)
voice = mixed - music
librosa.output.write_wav(VOICE_FILE_PATH, voice, sr)