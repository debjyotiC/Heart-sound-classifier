import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

features_mfcc = np.load("data/mfcc.npz", allow_pickle=True)
features_mfe = np.load("data/mfe.npz", allow_pickle=True)
features_lmfe = np.load("data/lmfe.npz", allow_pickle=True)

mfcc_data, mfcc_labels = features_mfcc['out_x'], features_mfcc['out_y']
mfe_data, mfe_labels = features_mfe['out_x'], features_mfe['out_y']
lmfe_data, lmfe_labels = features_lmfe['out_x'], features_lmfe['out_y']

pos = 2
num_frames = 150
num_ceps = 26

mfcc_data = mfcc_data[pos].reshape(num_frames, num_ceps).transpose()
mfe_data = mfe_data[pos].reshape(num_frames, num_ceps).transpose()
lmfe_data = lmfe_data[pos].reshape(num_frames, num_ceps).transpose()

fig, ax = plt.subplots(3, 1)
fig.tight_layout()
# MFCC plot
# ax[0].set_axis_off()
ax[0].imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[0].set_ylabel("No. of MFCCs")
ax[0].set_xlabel("No. of frames")
# ax[0].set_title('MFCC plot')

# MFE plot
# ax[1].set_axis_off()
ax[1].imshow(mfe_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[1].set_ylabel("No. of MFEs")
ax[1].set_xlabel("No. of frames")
# ax[1].set_title('MFE plot')

# # MFE plot
# ax[2].set_axis_off()
ax[2].imshow(lmfe_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[2].set_ylabel("No. of logMFEs")
ax[2].set_xlabel("No. of frames")
# ax[2].set_title('log MFE plot')

plt.savefig("mfcc-mfe-logmfe-plot.png", dpi=600)

plt.show()
