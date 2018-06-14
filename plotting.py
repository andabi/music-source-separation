from config import TrainConfig, ModelConfig
from data import Data

from librosa import amplitude_to_db, stft
from librosa.display import specshow
from preprocess import to_spectrogram, get_magnitude
from pylab import savefig

import matplotlib.pyplot as plt
import numpy as np


data = Data(TrainConfig.DATA_PATH)

mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(TrainConfig.SECONDS, TrainConfig.NUM_WAVFILE)
mixed_spec = to_spectrogram(mixed_wav)
mixed_mag = get_magnitude(mixed_spec)
src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)


sr = ModelConfig.SR
y = src1_wav[0]

def plot_wav_as_spec(wav, sr=ModelConfig.SR, s=0.5, path='foo.png'):
    """Plots a spectrogram

    Will save as foo.png in script directory

    Arguments:
        wav {array} -- audio data

    Keyword Arguments:
        sr {int} -- [sample rate] (default: {ModelConfig.SR})
        s {number} -- [size of plot] (default: {0.5})
        path {string} -- [where to save plot] (default: {foo.png})
    """

    plt.figure(figsize=(12*s, 8*s))
    D = amplitude_to_db(stft(y), ref=np.max)
    # plt.subplot(4, 2, 1)
    specshow(D, x_axis = 'time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    savefig(path)

plot_wav_as_spec(y)
