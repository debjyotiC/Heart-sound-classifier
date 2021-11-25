import numpy as np
import librosa
import matplotlib.pyplot as plt

path = '/Users/deb/Documents/heart-data/artifact/artifact__201012172012.wav'
sample_rate = 16000

signal, fs = librosa.load(path, sr=sample_rate)
signal = signal[0:int(1.5 * sample_rate)]

plt.plot(signal)

# data = np.load('data/mfcc-heart.npz', allow_pickle=True)  # load audio data
# x_data, y_data = data['out_x'], data['out_y']  # load into np arrays
#
# mfcc_calculated = x_data[1]
# print(y_data[1])
# # Plot MFCC
# fig = plt.figure()
# plt.imshow(mfcc_calculated, cmap='inferno', origin='lower')
plt.show()


