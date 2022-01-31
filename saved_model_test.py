import tensorflow as tf
import numpy as np

data_path = "data/mfcc.npz"
model_path = "saved_model/mfcc"

loaded_data = np.load(data_path)
x_data, y_data = loaded_data['out_x'], loaded_data['out_y']

classes_values = ["murmur", "normal"]
classes = len(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)
loaded_model = tf.keras.models.load_model(model_path)

prediction = loaded_model.predict(x_data)
right, wrong = 0, 0

for i in enumerate(prediction):
    actual = classes_values[np.argmax(y_data[i[0]])]
    predicted = classes_values[np.argmax(i[1])]
    if actual == predicted:
        print("Actual class is ", actual, "and predicted", predicted)
        right = right + 1
    else:
        print("WRONG!!! Actual class is ", actual, "but predicted", predicted)
        wrong = wrong + 1

print(f"Predicted {right} right and {wrong} wrong in total of {right+wrong}")

