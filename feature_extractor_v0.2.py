from os import listdir
from os.path import isdir, join
import speechpy as sp
import librosa
import numpy as np

np.set_printoptions(suppress=True)
dataset_path = '/Users/deb/Documents/heart-data'

sample_rate = 16000

frame_length = 0.02
frame_stride = 0.01
fft_size = 256

num_filter = 26
num_ceps = 26

pre_cof = 0.97
pre_shift = 1

num_frames = 148

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('exhaled')
all_targets.remove('artifact')
all_targets.remove('test')
print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


def calc_MFCC(path):
    signal, fs = librosa.load(path, sr=sample_rate)
    signal = signal[0: int(1.5 * fs)]
    signal_pre_emphasized = sp.processing.preemphasis(signal, cof=pre_cof, shift=pre_shift)
    mfccs = sp.feature.mfcc(signal_pre_emphasized, sampling_frequency=fs, frame_length=frame_length,
                            frame_stride=frame_stride, num_cepstral=num_ceps, num_filters=num_filter,
                            fft_length=fft_size)
    return mfccs


def calc_MFE(path):
    signal, fs = librosa.load(path, sr=sample_rate)
    signal = signal[0: int(1.5 * fs)]
    signal_pre_emphasized = sp.processing.preemphasis(signal, cof=pre_cof, shift=pre_shift)
    mfe, energy = sp.feature.mfe(signal_pre_emphasized, sampling_frequency=fs, frame_length=frame_length,
                                 frame_stride=frame_stride, num_filters=num_filter,
                                 fft_length=fft_size)
    return mfe


out_x_mfcc = []
out_y_mfcc = []

out_x_mfe = []
out_y_mfe = []

dropped, kept = 0, 0

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])
        print(full_path, folder)

        if not full_path.endswith('.wav'):
            continue

        mfcc_calculated = calc_MFCC(full_path)
        mfe_calculated = calc_MFE(full_path)
        if mfcc_calculated.shape[0] == num_frames and mfe_calculated.shape[0] == num_frames:
            out_x_mfcc.append(mfcc_calculated.flatten())
            out_y_mfcc.append(folder + 1)

            out_x_mfe.append(mfe_calculated.flatten())
            out_y_mfe.append(folder + 1)
            print("MFCC Shape: ", mfcc_calculated.shape)
            print("MFE Shape: ", mfe_calculated.shape)
            kept = kept + 1
        else:
            print('MFCC Dropped:', folder, mfcc_calculated.shape)
            print('MFE Dropped:', folder, mfe_calculated.shape)
            dropped = dropped + 1

data_mfcc_x = np.array(out_x_mfcc)
data_mfcc_y = np.array(out_y_mfcc)

data_mfe_x = np.array(out_x_mfe)
data_mfe_y = np.array(out_y_mfe)

print(f"Kept {kept} files and dropped {dropped} in total of {dropped+kept}")

print("MFCC Shape: ", data_mfcc_x.shape)
print("MFE Shape: ", data_mfe_x.shape)

print("saving NPZ file")
np.savez('data/mfcc.npz', out_x=data_mfcc_x, out_y=data_mfcc_y)  # store flattened MFCCs
np.savez('data/mfe.npz', out_x=data_mfe_x, out_y=data_mfe_y)  # store flattened MFEs
