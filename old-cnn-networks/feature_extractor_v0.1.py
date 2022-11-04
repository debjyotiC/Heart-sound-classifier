from os import listdir
from os.path import isdir, join
from python_speech_features import mfcc, fbank
import librosa
import numpy as np

np.set_printoptions(suppress=True)
dataset_path = '/Users/deb/Documents/heart-data'

sample_rate = 16000

frame_length = 0.256  # 0.256
frame_stride = 0.050  # 0.050
fft_size = 4096

num_filter = 26
num_ceps = 26
pre_emph = 0.97

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


def calc_mfcc(path):
    signal, fs = librosa.load(path, sr=sample_rate)
    signal = signal[0: int(1.5 * fs)]
    mfccs = mfcc(signal, samplerate=fs, winlen=frame_length, winstep=frame_stride, numcep=num_ceps, nfilt=num_filter,
                 nfft=fft_size, preemph=pre_emph, ceplifter=0, appendEnergy=False, winfunc=np.hanning)
    return mfccs.transpose()


def calc_mfe(path):
    signal, fs = librosa.load(path, sr=sample_rate)
    signal = signal[0: int(1.5 * fs)]
    mfe, energy = fbank(signal, samplerate=fs, winlen=frame_length, winstep=frame_stride, nfilt=num_filter,
                        nfft=fft_size, preemph=pre_emph, winfunc=np.hanning)
    return mfe.transpose()


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

        mfcc_calculated = calc_mfcc(full_path)
        mfe_calculated = calc_mfe(full_path)
        if mfcc_calculated.shape == (num_ceps, num_ceps) or mfe_calculated.shape == (num_filter, num_filter):
            out_x_mfcc.append(mfcc_calculated.flatten())
            out_y_mfcc.append(folder + 1)

            out_x_mfe.append(mfe_calculated.flatten())
            out_y_mfe.append(folder + 1)
            print("MFCC Shape: ", mfcc_calculated.shape)
            print("MFE Shape: ", mfe_calculated.shape)
            kept = kept + 1
        else:
            print(f"MFCC Dropped: {folder} {mfcc_calculated.shape}")
            print(f"MFE Dropped: {folder} {mfe_calculated.shape}")
            dropped = dropped + 1


data_mfcc_x = np.array(out_x_mfcc, dtype=object)
data_mfcc_y = np.array(out_y_mfcc, dtype=object)

data_mfe_x = np.array(out_x_mfe, dtype=object)
data_mfe_y = np.array(out_y_mfe, dtype=object)

print(f"Kept {kept} files and dropped {dropped} in total of {dropped+kept}")

print("MFCC Shape: ", data_mfcc_x.shape)
print("MFE Shape: ", data_mfe_x.shape)

np.savez('data/mfcc-flattened.npz', out_x=data_mfcc_x, out_y=data_mfcc_y)
np.savez('data/mfe-flattened.npz', out_x=data_mfe_x, out_y=data_mfe_y)
