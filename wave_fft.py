from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

wave_file = "data/test-360.wav"
fs, x = wavfile.read(wave_file)  # load the x
audio_data = x[:1000]
N = len(audio_data)
Ts = 1.0/fs

audio_fft = fft(audio_data)
audio_freq = fftfreq(N, Ts)[:N//2]
audio_mag = 2.0/N * np.abs(audio_fft[0:N//2])

print(f"Max frequency {audio_freq[np.argmax(audio_mag)]}")

# load Keras model
load_model = tf.keras.models.load_model('model/fft-clf')
test = np.vstack([audio_mag])
classes = load_model.predict(test)

if classes[0] > 0.5:
    print("is 1", classes[0])
else:
    print("is 0", classes[0])



