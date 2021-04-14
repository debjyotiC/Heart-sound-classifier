from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
wave_file = "data/tone-1.wav"
sample_rate, x = wavfile.read(wave_file)
N = len(x)

T = x[2]-x[1]   # time increment in each x

fft = abs(fft(x) * T)
freq = abs(fftfreq(N, d=T))

print(len(fft))

plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.semilogy(freq, fft)
plt.grid()
plt.show()

