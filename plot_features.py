import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

features_mfcc = np.load("data/mfcc-flattened.npz", allow_pickle=True)
features_mfe = np.load("data/mfe-flattened.npz", allow_pickle=True)

mfcc_data, mfcc_labels = features_mfcc['out_x'], features_mfcc['out_y']
mfe_data, mfe_labels = features_mfe['out_x'], features_mfe['out_y']

mfcc_data_0 = mfcc_data[0].reshape(26, 26)
mfe_data_0 = mfe_data[0].reshape(26, 26)


fig, ax = plt.subplots(1, 2)
# MFCC plot
ax[0].set_axis_off()
ax[0].imshow(mfcc_data_0, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[0].set_title('MFCC plot')

# MFE plot
ax[1].set_axis_off()
ax[1].imshow(mfe_data_0, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax[1].set_title('MFE plot')
plt.show()
