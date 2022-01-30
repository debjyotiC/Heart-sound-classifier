from os import listdir
from os.path import isdir, join
import speechpy as sp
import librosa
import numpy as np

np.set_printoptions(suppress=True)
dataset_path = '/Users/deb/Documents/heart-data'

sample_rate = 16000
num_mfcc = 26

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
    signal = signal[0:int(1.46 * sample_rate)]  # keep first 3 sec of the audio data
    mfccs = sp.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.256, frame_stride=0.0446, num_cepstral=27,
                            num_filters=26, fft_length=4096)
    mfccs = np.array(mfccs)
    return mfccs.transpose()


def calc_MFE(path):
    signal, fs = librosa.load(path, sr=sample_rate)
    signal = signal[0:int(1.46 * sample_rate)]  # keep first 3 sec of the audio data
    mfe, energy = sp.feature.mfe(signal, sampling_frequency=fs, frame_length=0.256, frame_stride=0.0446,
                                 num_filters=26, fft_length=4096)
    mfes = np.array(mfe)
    return mfes.transpose()


out_x = []
out_y = []

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])
        print(full_path, folder)

        if not full_path.endswith('.wav'):
            continue

        mfcc_calculated = calc_MFCC(full_path)
        if mfcc_calculated.shape == (num_mfcc, num_mfcc):
            out_x.append(mfcc_calculated)
            out_y.append(folder + 1)
            print(mfcc_calculated.shape)
        else:
            print('Dropped:', folder, mfcc_calculated.shape)

data_x = np.array(out_x)
data_y = np.array(out_y)

print(data_x.shape)

# np.savez('data/mfcc-flattened.npz', out_x=data_x, out_y=data_y)
