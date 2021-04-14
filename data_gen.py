from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Number of sample points
N = 1000

# sample spacing
T = 1.0 / 800.0

x = np.linspace(0.0, N*T, N, endpoint=False)
y_1 = 10 * np.sin(50.0 * 2.0*np.pi*x)
y_2 = 10 * np.sin(70.0 * 2.0*np.pi*x)

y_3 = 10 * np.sin(350.0 * 2.0*np.pi*x)
y_4 = 10 * np.sin(370.0 * 2.0*np.pi*x)

y_test_1 = 6.5 * np.sin(60.0 * 2.0*np.pi*x)
y_test_2 = 6.5 * np.sin(360.0 * 2.0*np.pi*x)

yf_1 = fft(y_1)
yf_2 = fft(y_2)

yf_3 = fft(y_3)
yf_4 = fft(y_4)
yf_test_1 = fft(y_test_1)
yf_test_2 = fft(y_test_2)

xf = fftfreq(N, T)[:N//2]

mag_1 = 2.0/N * np.abs(yf_1[0:N//2])
mag_2 = 2.0/N * np.abs(yf_2[0:N//2])

mag_3 = 2.0/N * np.abs(yf_3[0:N//2])
mag_4 = 2.0/N * np.abs(yf_4[0:N//2])

mag_test_1 = 2.0/N * np.abs(yf_test_1[0:N//2])
mag_test_2 = 2.0/N * np.abs(yf_test_2[0:N//2])

plt.plot(xf, mag_1)
plt.plot(xf, mag_2)
plt.plot(xf, mag_4)
plt.plot(xf, mag_3)
plt.plot(xf, mag_test_1)
plt.plot(xf, mag_test_2)
plt.grid()
plt.show()

# save x train
# values_1 = {'Frequency': xf, 'Mag_1': mag_1, 'Mag_2': mag_2, 'Mag_3': mag_3, 'Mag_4': mag_4}
# df_w_1 = pd.DataFrame(values_1, columns=['Frequency', 'Mag_1', 'Mag_2', 'Mag_3', 'Mag_4'])
# df_w_1.to_csv("x/fft-x-3.csv", index=False, header=True)


# save x test
# values_1 = {'Frequency': xf, 'Mag_1': mag_test_1, 'Mag_2': mag_test_2}
# df_w_1 = pd.DataFrame(values_1, columns=['Frequency', 'Mag_1', 'Mag_2'])
# df_w_1.to_csv("x/fft-x-test.csv", index=False, header=True)

