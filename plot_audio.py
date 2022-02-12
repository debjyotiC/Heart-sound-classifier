import numpy as np
import speechpy as sp
import matplotlib.pyplot as plt
loaded_file = np.load("data/mfe.npz", allow_pickle=True)

x_data, y_data = loaded_file["out_x"], loaded_file["out_y"]

mfcc_data = x_data[0].reshape(26, 26)

plt.imshow(mfcc_data, cmap='inferno', origin='lower')
plt.show()
