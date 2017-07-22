# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

from pydub import AudioSegment
import os


# Necessary libraries: ffmpeg, libav
def rewrite_mp3_to_wav(source_path, target_path):
    basepath, filename = os.path.split(source_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(source_path).export(target_path, format='wav')