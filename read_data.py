import numpy as np
import pandas as pd
import glob
import os, fnmatch
import librosa
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

INPUT_DIR = "C:\\Users\\Deb\\Documents\\heart-data"
SAMPLE_RATE = 16000
MAX_SOUND_CLIP_DURATION = 12  # sec

set_a = pd.read_csv(INPUT_DIR + "/set_a.csv")
set_b = pd.read_csv(INPUT_DIR + "/set_b.csv")
frames = [set_a, set_b]
data_ab = pd.concat(frames)


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


# get audio data without padding highest quality audio
def load_file_data_without_change(folder, file_names, duration=3, sr=16000):
    input_length = sr * duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # extract normalized mfcc feature from data
            # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
            data.append(X)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        # feature = np.array(mfccs).reshape([-1,1])
    return data


# get audio data with a fix padding may also chop off some file
def load_file_data(folder, file_names, duration=12, sr=16000):
    global y
    input_length = sr * duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if round(dur) < duration:
                print("fixing audio lenght :", file_name)
                y = librosa.util.fix_length(X, input_length)
                # normalized raw audio
                y = audio_norm(y)
            # extract normalized mfcc feature from data
            # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
            data.append(y)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            # feature = np.array(mfccs).reshape([-1,1])
    return data


# Map label text to integer
CLASSES = ['artifact', 'murmur', 'normal']
# {'artifact': 0, 'murmur': 1, 'normal': 3}
NB_CLASSES = len(CLASSES)

# Map integer value to text labels
label_to_int = {k: v for v, k in enumerate(CLASSES)}
print(label_to_int)
print(" ")
# map integer to label text
int_to_label = {v: k for k, v in label_to_int.items()}
print(int_to_label)

A_folder = INPUT_DIR + '/set_a/'
# set-a
A_artifact_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'artifact*.wav')
A_artifact_sounds = load_file_data(folder=A_folder, file_names=A_artifact_files, duration=MAX_SOUND_CLIP_DURATION)
A_artifact_labels = [0 for items in A_artifact_files]

A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'normal*.wav')
A_normal_sounds = load_file_data(folder=A_folder, file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)
A_normal_labels = [2 for items in A_normal_sounds]

A_extrahls_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'extrahls*.wav')
A_extrahls_sounds = load_file_data(folder=A_folder, file_names=A_extrahls_files, duration=MAX_SOUND_CLIP_DURATION)
A_extrahls_labels = [1 for items in A_extrahls_sounds]

A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'murmur*.wav')
A_murmur_sounds = load_file_data(folder=A_folder, file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
A_murmur_labels = [1 for items in A_murmur_files]

# test files
A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_a'), 'Aunlabelledtest*.wav')
A_unlabelledtest_sounds = load_file_data(folder=A_folder, file_names=A_unlabelledtest_files,
                                         duration=MAX_SOUND_CLIP_DURATION)
A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

print("loaded dataset-a")

# load dataset-b, keep them separate for testing purpose
B_folder = INPUT_DIR + '/set_b/'
# set-b
B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'normal*.wav')  # include noisy files
B_normal_sounds = load_file_data(folder=B_folder, file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)
B_normal_labels = [2 for items in B_normal_sounds]

B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'murmur*.wav')  # include noisy files
B_murmur_sounds = load_file_data(folder=B_folder, file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
B_murmur_labels = [1 for items in B_murmur_files]

B_extrastole_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'extrastole*.wav')
B_extrastole_sounds = load_file_data(folder=B_folder, file_names=B_extrastole_files, duration=MAX_SOUND_CLIP_DURATION)
B_extrastole_labels = [1 for items in B_extrastole_files]

# test files
B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + '/set_b'), 'Bunlabelledtest*.wav')
B_unlabelledtest_sounds = load_file_data(folder=B_folder, file_names=B_unlabelledtest_files,
                                         duration=MAX_SOUND_CLIP_DURATION)
B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]
print("loaded dataset-b")

# combine set-a and set-b
x_data = np.concatenate((A_artifact_sounds, A_normal_sounds, A_extrahls_sounds, A_murmur_sounds,
                         B_normal_sounds, B_murmur_sounds, B_extrastole_sounds))

y_data = np.concatenate((A_artifact_labels, A_normal_labels, A_extrahls_labels, A_murmur_labels,
                         B_normal_labels, B_murmur_labels, B_extrastole_labels))

test_x = np.concatenate((A_unlabelledtest_sounds, B_unlabelledtest_sounds))
test_y = np.concatenate((A_unlabelledtest_labels, B_unlabelledtest_labels))

print("combined training data record: ", len(y_data), len(test_y))


np.savez('data/audio-data.npz', out_x=x_data, out_y=y_data)
