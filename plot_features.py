import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

features_mfcc = np.load("data/mfcc.npz", allow_pickle=True)
features_mfe = np.load("data/mfe.npz", allow_pickle=True)

mfcc_data, mfcc_labels = features_mfcc['out_x'], features_mfcc['out_y']
mfe_data, mfe_labels = features_mfe['out_x'], features_mfe['out_y']

pos = 10
num_frames = 148
num_ceps = 26

mfcc_data = mfcc_data[pos].reshape(num_frames, num_ceps).transpose()
mfe_data = mfe_data[pos].reshape(num_frames, num_ceps).transpose()


fig, ax = plt.subplots(2, 1)
# MFCC plot
ax[0].set_axis_off()
ax[0].imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[0].set_title('MFCC plot')

# MFE plot
ax[1].set_axis_off()
ax[1].imshow(mfe_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[1].set_title('MFE plot')
plt.savefig("test.png")
plt.show()
