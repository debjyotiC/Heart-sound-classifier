import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd

path = '/Users/deb/Documents/heart-data/normal/normal__103_1305031931979_B.wav'
sample_rate = 16000

signal, fs = librosa.load(path, sr=sample_rate)
signal = signal[0:int(1.5 * sample_rate)]

plt.plot(signal, '-', label="Normal")
plt.xlabel("second")
plt.ylabel("magnitude")
plt.grid(True)
plt.legend(loc="upper left")
plt.show()

# plt.plot(signal)
# plt.show()
# data = np.load('data//mfcc-murmur-normal.npz', allow_pickle=True)  # load audio data
# x_data, y_data = data['out_x'], data['out_y']  # load into np arrays
#
# mfcc_calculated = x_data[32]
# print(y_data[32])
# # plt.plot(signal)
#
# # Plot MFCC
# fig = plt.figure()
# plt.imshow(mfcc_calculated, cmap='inferno', origin='lower')
# plt.show()
