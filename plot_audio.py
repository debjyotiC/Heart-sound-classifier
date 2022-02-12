import numpy as np
import librosa
import matplotlib.pyplot as plt


time = 2.0
alpha = 0.97

file_path = "/Users/deb/Documents/heart-data/murmur/murmur__112_1306243000964_A.wav"

signal, fs = librosa.load(file_path, sr=None)
print(fs)
signal = signal[0: int(time * fs)]


emphasized_signal = np.append(signal[0], signal[1:]-alpha*signal[:-1])

fig, ax = plt.subplots(3, 1)
fig.tight_layout()
ax[0].plot(signal)
ax[0].set_ylabel("Amplitude")
ax[0].set_xlabel("Time (millisecond)")
ax[0].grid(True)

ax[1].plot(emphasized_signal)
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("Time (millisecond)")
ax[1].grid(True)
ax[2].set_axis_off()
plt.savefig("signal.png", dpi=600)
plt.show()