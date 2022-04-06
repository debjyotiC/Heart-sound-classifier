import numpy as np
import ctypes

features_mfcc = np.load("data/mfcc.npz", allow_pickle=True)

mfcc_data, mfcc_labels = features_mfcc['out_x'], features_mfcc['out_y']

pos = 0
data = mfcc_data[pos]

# print(f"Overall all length is {len(mfcc_data)}")
# print(f"Length of individual data is {len(data)}")
data_rounded = np.round(data, 2)
data_inlist = data_rounded.tolist()



print(data.shape)

