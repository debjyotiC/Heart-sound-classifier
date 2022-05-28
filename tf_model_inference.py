import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

DEBUG = False
model_type = "mfe"
model_path = f"saved_tflite_model/{model_type}_default.tflite"
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
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for data in enumerate(x_data):
    input_data = data[1].astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], np.vstack([input_data]))
    interpreter.invoke()

    classes = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(classes)

    predicted = classes_values[np.argmax(results)]
    actual = classes_values[np.argmax(y_data[data[0]])]

    label_actual.append(actual)
    label_predicted.append(predicted)


results = confusion_matrix(label_actual, label_predicted)
acc = accuracy_score(label_actual, label_predicted)

print('Accuracy Score :', acc)
print(f'Classification report for {model_type.upper()} model: ')
print(classification_report(label_actual, label_predicted))


ax = plt.subplot()
sns.heatmap(results, annot=True, ax=ax, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Confusion Matrix for {model_type.upper()} TFLite model accuracy {round(acc, 2)}')
ax.xaxis.set_ticklabels(classes_values)
ax.yaxis.set_ticklabels(classes_values)
plt.savefig(f'images/tflite_confusion_matrix_{model_type}.png', dpi=600)
plt.show()
