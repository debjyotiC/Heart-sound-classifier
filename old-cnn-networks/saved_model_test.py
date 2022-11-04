import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

DEBUG = True
model_type = "mfcc"
data_path = f"data/{model_type}.npz"
model_path = f"saved_model/{model_type}"

loaded_data = np.load(data_path)
x_data, y_data = loaded_data['out_x'], loaded_data['out_y']

label_actual, label_predicted = [], []
classes_values = ["murmur", "normal"]
classes = len(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)
loaded_model = tf.keras.models.load_model(model_path)

if DEBUG:
    print(x_data.shape)

prediction = loaded_model.predict(x_data)
right, wrong = 0, 0

for i in enumerate(prediction):
    actual = classes_values[np.argmax(y_data[i[0]])]
    predicted = classes_values[np.argmax(i[1])]
    if actual == predicted:
        if DEBUG:
            print("Actual class is", actual, "and predicted", predicted)
        right = right + 1
    else:
        if DEBUG:
            print("WRONG!!! Actual class is", actual, "but predicted", predicted)
        wrong = wrong + 1
    label_actual.append(actual)
    label_predicted.append(predicted)

if DEBUG:
    print(f"Loaded {model_path.split('/')[1]} model")
    print(f"Predicted {right} right and {wrong} wrong in total of {right+wrong}")

results = confusion_matrix(label_actual, label_predicted)

print('Accuracy Score :', accuracy_score(label_actual, label_predicted))
print(f'Classification report for {model_type.upper()} model: ')
print(classification_report(label_actual, label_predicted))


ax = plt.subplot()
sns.heatmap(results, annot=True, ax=ax, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Confusion Matrix for {model_type.upper()} model')
ax.xaxis.set_ticklabels(classes_values)
ax.yaxis.set_ticklabels(classes_values)
plt.savefig(f'images/confusion_matrix_{model_type}.png', dpi=600)
plt.show()
