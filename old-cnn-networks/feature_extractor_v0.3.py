from os import listdir
from os.path import isdir, join
import speechpy as sp
import librosa
from librosa.util import fix_length
import numpy as np

np.set_printoptions(suppress=True)
dataset_path = '/Users/deb/Documents/heart-data'
SAVE = True

frame_length = 0.5
frame_stride = 0.01
fft_size = 256

num_filter = 26
num_ceps = 26

pre_cof = 0.97
pre_shift = 1

time = 2.0
limit = 2
num_frames = 150

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('other')
all_targets.remove('noisy')
print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


def calc_MFCC(path):
    signal, fs = librosa.load(path, sr=None)
    signal = fix_length(signal, size=limit * fs)
    signal_pre_emphasized = sp.processing.preemphasis(signal, cof=pre_cof, shift=pre_shift)
    mfccs = sp.feature.mfcc(signal_pre_emphasized, sampling_frequency=fs, frame_length=frame_length,
                            frame_stride=frame_stride, num_cepstral=num_ceps, num_filters=num_filter,
                            fft_length=fft_size)
    return mfccs


out_x_mfcc = []
out_y_mfcc = []

dropped, kept = 0, 0

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])
        print(full_path, folder)

        if not full_path.endswith('.wav'):
            continue

        mfcc_calculated = calc_MFCC(full_path)

        if mfcc_calculated.shape[0] == num_frames:
            out_x_mfcc.append(mfcc_calculated.flatten())
            out_y_mfcc.append(folder + 1)

            print("MFCC Shape: ", mfcc_calculated.shape)
            kept = kept + 1
        else:
            print('MFCC Dropped:', folder, mfcc_calculated.shape)
            dropped = dropped + 1

data_mfcc_x = np.array(out_x_mfcc)
data_mfcc_y = np.array(out_y_mfcc)

data_mfcc_x_int = np.array(out_x_mfcc, dtype="int8")

print(f"Kept {kept} files and dropped {dropped} in total of {dropped + kept}")

print("MFCC Shape: ", data_mfcc_x.shape)

if SAVE:
    np.savez('data/mfcc.npz', out_x=data_mfcc_x, out_y=data_mfcc_y)  # store flattened MFCCs
    np.savez('data/mfcc_int8.npz', out_x=data_mfcc_x_int, out_y=data_mfcc_y)  # store flattened MFCCs

    print("saved NPZ file")
else:
    pass
