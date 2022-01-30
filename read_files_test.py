import python_speech_features as psf
import librosa
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

sample_rate = 16000
num_mfcc = 26

path_murmur = "C:/Users/Deb/Documents/heart-data/murmur/murmur__122_1306325762831_C.wav"
path_normal = "C:/Users/Deb/Documents/heart-data/normal/normal__103_1305031931979_B.wav"
path_unlabeled_test = "C:/Users/Deb/Documents/heart-data/exhaled/extrahls__201101070953.wav"


def calc_mfcc(file_path, s_rate):
    signal, fs = librosa.load(file_path, sr=s_rate)
    signal = signal[0:int(1.5 * sample_rate)]  # keep first 3 sec of the audio data
    mfccs = psf.base.mfcc(signal, samplerate=fs, winlen=0.256, winstep=0.050, numcep=num_mfcc, nfilt=26,
                          nfft=4096, preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=np.hanning)
    return mfccs.transpose()


mfcc_got = calc_mfcc(file_path=path_murmur, s_rate=sample_rate)
mfcc_got = mfcc_got.reshape(1, mfcc_got.shape[0], mfcc_got.shape[1], 1)

loaded_model = tf.keras.models.load_model("saved_model/mfcc")
classes = loaded_model.predict(mfcc_got)
print(classes)

if classes[0][0] > 0.5:
    print("Class 1")
else:
    print("Class 0")
