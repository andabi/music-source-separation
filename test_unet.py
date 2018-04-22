#from preprocess_data import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from kapre_helpers import *
from data import Data
from config import TrainConfig, ModelConfig, EvalConfig
from preprocess import *
from utils import *
# from model import Model
# from unet_definity import unet

# Init Params
nwins = (ModelConfig.SR * TrainConfig.SECONDS) / ModelConfig.L_FRAME
nbins = int(ModelConfig.L_FRAME * nwins)

# Load Data
data = Data(TrainConfig.DATA_PATH)
mix, music, voc, wavfiles = data.next_wavs(sec=TrainConfig.SECONDS,
                                           size=TrainConfig.NUM_WAVFILE)

# Export Input File
librosa.output.write_wav(path='input.wav', y=voc.T, sr=ModelConfig.SR, norm=True)

# Batch to Spectogram
x = voc
x = to_spectrogram(x)
x_phase = get_phase(x)
x = get_magnitude(x)
print('x before batch {}'.format(x.shape))
batch_size = x.shape[0]

# Convert to unet shape, eg not 1, 513 x 513
x = x[:,0:512,0:512]

# Export wav to check if spectogram was ok
nd_array_to_txt(filename='librosa_spec', data=x)
out_wav = to_wav(x, x_phase[:,0:512,0:512])
write_wav(data = out_wav.T, path = 'out_before')

# Export spectogram images
for i in range(batch_size):
    print('spectogram-before-model-' + str(i) + '.png')
    plot_spect(x, name='spectogram-before' + str(i) + '.png')

# #################   testing
    # print('voc shape {}'.format(voc.shape))
    # inputs_mel = Input(voc.shape)
    # melspec = Melspectrogram(
    #     input_shape=voc.shape,
    #     n_dft=ModelConfig.L_FRAME,
    #     n_hop=ModelConfig.L_HOP,
    #     n_mels=ModelConfig.N_MELS,
    #     sr=ModelConfig.SR,
    #     fmin=ModelConfig.F_MIN,
    #     fmax=ModelConfig.SR / 2,
    #     trainable_fb=False,
    #     trainable_kernel=False)(inputs_mel)

    # Make Kapre Melspec
    # model_pre = Model(inputs=inputs_mel, outputs=melspec)
    # x_mel = model_pre.predict(voc[np.newaxis,:], batch_size=x.shape[0], verbose=1)

    # # Print shape help
    print('Batch size: {}'.format(batch_size))
    print('x specto / input shape: {}'.format(x.shape))
    # print('x mel shape / input shape: {}'.format(x_mel.shape))

    # # Export Kapre mel and wav
    # x_mel_temp = x_mel[:,:,:,0]
    # nd_array_to_txt(filename='kapre_mel', data=x_mel_temp)
    # plot_spect(x_mel_temp, name='spectogram-kapre-before' + str(i) + '.png')
    # mel_wav = to_wav_from_spec(x_mel_temp)
    # librosa.output.write_wav(path='out_mel.wav', y=mel_wav.T, sr=ModelConfig.SR, norm=True)
    # # Make it 512 cols and rows
    # #x_mel = x_mel[:,0:512,:,0]

    ##############

# Conform to unet shape
x = x[:, :, :, np.newaxis]

print(x.ndim)
print('X newaxis shape {}'.format(x.shape))

# Propagate Unet
input_shape = (x.shape[1], x.shape[2], 1)
inputs = Input(input_shape)
outputs = unet(inputs=inputs)
model = Model(inputs=inputs, outputs=outputs)
y_hat = model.predict(x, batch_size=batch_size, verbose=1)

# Print output shapes
print('y_hat shape {}'.format(y_hat.shape))
print('len yhat {}'.format(len(y_hat)))

# get rid of weird unet shape
y_hat = y_hat[:,:,:,0]

# Export Spectogram images after model
for i in range(batch_size):
    print('spectogram-after-unet_' + str(i) + '.png')
    plot_spect(y_hat, name='spectogram-after-unet_' + str(i) + '.png')

# Convert from spectogram to wav
out_wav = to_wav(mag = y_hat, phase=x_phase[:,0:512,0:512])
# out_wav = to_wav_from_spec(y_hat[:,:,:,0])

# Export wav after model
librosa.output.write_wav(path='out_after_librosa.wav', y=out_wav.T, sr=ModelConfig.SR, norm=True)
write_wav(data = out_wav.T, path = 'out_after_writewavfunc')

# Print shapes
print('in wav shape {}'.format(voc.shape))
print('out_wav shape {}'.format(out_wav.shape))
























# print(nwins)

# print(music.shape)


# x = Melspectrogram(
#     input_shape=(, 44100), # 1-sec stereo input
#     n_dft=512, n_hop=256, n_mels=128, sr=sr,
#     fmin=0.0, fmax=sr/2, return_decibel=False,
#     trainable_fb=False, trainable_kernel=False)(music)

# Normalization2D(str_axis=’freq’)

# AdditiveNoise(power=0.2)
# and more layers for model hereafter




# for some reason we need this for Melspecto to work ...
    # testshape = to_spectrogram(mix)
    # input_shape = mix.shape

# x = voc[np.newaxis, :]
# batch_size = x.shape[0]
# inputs_specto = Input(input_shape)

# output shape of melspec will be nbins / n_hop
    # we want it to be 512, 512
    # melspec = Melspectrogram(
    #     input_shape=(input_shape),
    #     n_dft = ModelConfig.L_FRAME,
    #     n_hop = ModelConfig.L_HOP,
    #     n_mels= ModelConfig.N_MELS,
    #     sr = ModelConfig.SR,
    #     fmin = ModelConfig.F_MIN,
    #     fmax = ModelConfig.SR / 2,
    #     trainable_fb=False,
    #     trainable_kernel=False)(inputs_specto)

# specto = Spectrogram(
    #     n_dft=ModelConfig.L_FRAME,
    #     n_hop=ModelConfig.L_HOP,
    #     input_shape=input_shape,
    #     return_decibel_spectrogram=True,
    #     power_spectrogram=2.0,
    #     trainable_kernel=False,
    #     name='static_stft')(inputs_specto)

