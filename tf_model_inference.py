import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

DEBUG = False
model_type = "mfcc_int8"
model_path = f"saved_tflite_model/{model_type}.tflite"
data_path = f"data/{model_type}.npz"

loaded_data = np.load(data_path)
x_data, y_data = loaded_data['out_x'], loaded_data['out_y']

label_actual, label_predicted = [], []
classes_values = ["murmur", "normal"]
classes = len(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_data = np.expand_dims(x_data, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
classes = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(classes)
print(results)
