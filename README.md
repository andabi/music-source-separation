# Deep Neural Network for Music Source Separation in Tensorflow
## Intro
Recently, deep neural networks have been used in numorous fields and improved quality of many tasks in the fields. 
Applying deep neural nets to MIR(Music Information Retrieval) tasks also gave us quantum quality improvement. 
In this project, I implemented a neural network model for music source separation in Tensorflow.
Music source separation is the task to separate vocal sound from music such as k-pop.
## Experiments
I used Posen's deep recurrent neural network(RNN) model [2, 3]. But I used iKala dataset introduced by [1] instead of using MIR-1K dataset which is open to researchers.
3 RNN layers + 2 dense layer + 2 time-frequency masking layer
* B = batch size
* S = sequence length
* F = number of frequencies
# \[Related Paper\] Singing-Voice Separation From Monaural Recordings Using Deep Recurrent Neural Networks (2014)
## Proposed Methods
### Overall process
* Waveform of a music(mixed wav) is transformed to magnitude and phase spectra.
* It could be converted by Short-Time Fourier Transformation(STFT).
* I only make use of the magnitude as input feature of the RNN layer.
* I get the estimated magnitude spectra of each sources as outputs of the model.
* The estimated spectra is transformed to waveform of each sources by ISTFT.
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/overall.png" width="75%"></p>

### Model
* RNN layers (3 layers)
* Dense layer
  * each layer for each source
* Time-frequency masking layer
  * each layer for each source
  * regularize sum of outputs of each dense layer for each (time, frequency) to be inputs (mixed)
  * no non-linearity
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/model.png" width="75%"></p>

### Loss
* Mean squared error(MSE) or KL divergence between estimated magnitude and ground true are used as the loss function.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/mse.png" height="30px"></p>

<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/kl.png" height="30px"></p>

* Further, to prevent different sources to get similar each other, 'discrimination' term is considered additionally.
  * The discrimination weight(r) should be carefully chosen because it causes ignoring the first term when training(large r (e.g. r >= 1) makes the result bad)
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/disc_mse.png" height="30px"></p>

<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/disc_kl.png" height="30px"></p>


## Experiments
### Settings
#### [MIR-1K dataset](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)
* 1000 song clip with a sample rate of 16KHz, with duration from 4 to 13 secs.
* extracted from 110 Karaoke songs performed by both male and female amateurs.
* singing voice and background music in differenct channels.
#### Data augmentation
data augmentation: circularly shift the singing voice and mix them with the background music.
1024 points STFT with 50% overlap
L-BFGS optimizer
#### Evaluation Metric
[BSS-EVAL 3.0 metrics](https://hal.inria.fr/inria-00544230/document) are used.
v^ = estimated singing voice, v = ground truth singing voice, m = ground truth background music
x = the mixture
* Source to Distortion Ratio (SDR), GSDR: how much similar the estimated sound is with the original audio from same source?
* Source to Interferences Ratio (SIR), GSIR: how much discriminative the estimated sound is with the audio from different sources?
* Sources to Artifacts Ratio (SAR), GSAR:
* NSDR(Normalized SDR), GNSDR: SDR improvement between the estimated singing voice and the mixture.
  * SDR(v^, v) - SDR(x, v)
### Results
* The proposed neural network models achieve 2.30-2.48 dB GNSDR gain, 4.32-5.42 dB GSIR gain with similar GSAR performance, compared with conventional approaches. (quantum jump!!!)
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/result3.png" width="50%"></p>

* Concatenating neighboring 1 frame provides better results.
We can make a assumption that more sufficient information than single frame provides more hint to the neural net.
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/result1.png" width="50%"></p>

* The RNN-based models, in fact, do not make any plausible improvement comparing with DNN.
But discriminative training with carefully chosen weight(r) provides a bit better performance in the experiments.
<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/result1.png" width="50%"></p>

<p align="center"><img src="https://raw.githubusercontent.com/andabi/music-source-separation/master/materials/result4.png" width="100%"></p>

# Music Signal Processing Using Vector Product Neural Networks (2017)
# References
1. Zhe-Cheng Fan, Tak-Shing T. Chan, Yi-Hsuan Yang, and Jyh-Shing R. Jang, "[Music Signal Processing Using Vector Product
Neural Networks](http://mac.citi.sinica.edu.tw/~yang/pub/fan17dlm.pdf)", Proc. of the First Int. Workshop on Deep Learning and Music joint with IJCNN, May, 2017
2. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation](http://paris.cs.illinois.edu/pubs/huang-ismir2014.pdf)", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 12, pp. 2136–2147, Dec. 2015
3. P.-S. Huang, M. Kim, M. Hasegawa-Johnson, P. Smaragdis, "[Singing-Voice Separation From Monaural Recordings Using Deep Recurrent Neural Networks](https://posenhuang.github.io/papers/DRNN_ISMIR2014.pdf)" in International Society for Music Information Retrieval Conference (ISMIR) 2014.