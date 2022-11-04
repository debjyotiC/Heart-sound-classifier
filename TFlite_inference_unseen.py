import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math

feature_type = "lmfe"
type_of_quantization = "Default"

model_path = f"saved_tflite_model/{feature_type}_{type_of_quantization}.tflite"

data = np.load(f"data/{feature_type}.npz", allow_pickle=True)
x_data, y_data = data['out_x'], data['out_y']

label_actual, label_predicted = [], []
classes_values = ["murmur", "normal"]
classes = len(classes_values)

# train is now 75% of the entire data set
# validation is now 20% of the initial data set
# test is now 10% of the initial data set
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10


y_data = tf.keras.utils.to_categorical(y_data - 1, classes)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio, random_state=0,
                                                    stratify=y_data)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=0, stratify=y_test)

x_val = x_val.astype(np.float32)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for data in enumerate(x_val):
    interpreter.set_tensor(input_details[0]['index'], np.vstack([data[1]]))
    interpreter.invoke()

    classes = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(classes)
    predicted = classes_values[np.argmax(results)]
    actual = classes_values[np.argmax(y_data[data[0]])]

    label_actual.append(actual)
    label_predicted.append(predicted)

tn, fp, fn, tp = confusion_matrix(label_actual, label_predicted).ravel()
acc = accuracy_score(label_actual, label_predicted)
report = classification_report(label_actual, label_predicted, output_dict=True)
results = confusion_matrix(label_actual, label_predicted)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

x = sensitivity / (1 - sensitivity)
y = specificity / (1 - specificity)

youdens_index = sensitivity - (1 - specificity)
discriminant_power = (math.sqrt(3) / math.pi) * (math.log(x) + math.log(y))

print(f"For feature type: {feature_type}")
print(f"Accuracy Score: {acc}")
print(f"F1: {report['weighted avg']['f1-score']}")
print(f"Precision: {precision}")
print(f"Youden's Index: {youdens_index}")
print(f"Discriminant Power: {discriminant_power}")



