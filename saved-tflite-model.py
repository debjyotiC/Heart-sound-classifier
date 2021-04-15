import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv("data/fft-data-test.csv").dropna().reset_index(drop=True)

x_test = df.drop(columns=['Frequency']).to_numpy(dtype='float32')
data_selector = 0

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test = np.vstack([x_test[data_selector]])
interpreter.set_tensor(input_details[0]['index'], test)

interpreter.invoke()

classes = interpreter.get_tensor(output_details[0]['index'])

if classes[0] > 0.5:
    print("is 1", classes[0])
else:
    print("is 0", classes[0])
