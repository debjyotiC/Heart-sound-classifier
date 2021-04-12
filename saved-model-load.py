import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("data/fft-data-test.csv").dropna().reset_index(drop=True)

x_test = df.drop(columns=['Frequency']).to_numpy(dtype='float64')
data_selector = 0

# load Keras model
load_model = tf.keras.models.load_model('model/fft-clf')
test = np.vstack([x_test[data_selector]])
classes = load_model.predict(test)
print(classes)
if classes[0] > 0.5:
    print("is 1")
else:
    print("is 0")
