import numpy as np


features_mfcc = np.load("data/mfcc.npz", allow_pickle=True)

mfcc_data, mfcc_labels = features_mfcc['out_x'], features_mfcc['out_y']

pos = 0
classes_values = ["murmur", "normal"]
data = mfcc_data[pos]
label = classes_values[mfcc_labels[pos]-1]

# print(f"Overall all length is {len(mfcc_data)}")
# print(f"Length of individual data is {len(data)}")
data_rounded = np.round(data, 2)
data_inlist = data_rounded.tolist()

for i in data_rounded:
    print(i, end=',')
print(label)
