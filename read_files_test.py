import speechpy as sp
import librosa
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

frame_length = 0.5
frame_stride = 0.01
fft_size = 256

num_filter = 26
num_ceps = 26

pre_cof = 0.97
pre_shift = 1

time = 2.0
num_frames = 150

path_murmur = "/Users/deb/Documents/heart-data/murmur/murmur__122_1306325762831_C.wav"


def calc_MFCC(path):
    signal, fs = librosa.load(path, sr=None)
    signal = signal[0: int(time * fs)]
    signal_pre_emphasized = sp.processing.preemphasis(signal, cof=pre_cof, shift=pre_shift)
    mfccs = sp.feature.mfcc(signal_pre_emphasized, sampling_frequency=fs, frame_length=frame_length,
                            frame_stride=frame_stride, num_cepstral=num_ceps, num_filters=num_filter,
                            fft_length=fft_size)
    return mfccs.flatten()


mfcc_got = calc_MFCC(path=path_murmur)
print(mfcc_got.shape)

loaded_model = tf.keras.models.load_model("saved_model/mfcc")
classes = loaded_model.predict(mfcc_got)
print(classes)


