import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data_type = "mfe"
data = np.load(f'data/{data_type}_2_test.npz', allow_pickle=True)

predicted_label, actual_label = data['out_x'], data['out_y']
classes_values = ["murmur", "normal"]
label_predicted = np.argmax(predicted_label, axis=1)
label_actual = np.argmax(actual_label, axis=1)

# print(predicted_label)
# print(actual_label)

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

print(f"For feature type: {data_type}")
print(f"Accuracy Score: {acc}")
print(f"F1: {report['weighted avg']['f1-score']}")
print(f"Precision: {precision}")
print(f"Youden's Index: {youdens_index}")
print(f"Discriminant Power: {discriminant_power}")

ax = plt.subplot()
sns.heatmap(results, annot=True, annot_kws={"size": 20}, ax=ax, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=12)
ax.set_ylabel('True labels', fontsize=12)
# ax.set_title(f'Confusion Matrix for {data_type.upper()} TFLite model accuracy {round(acc, 2)}')
ax.xaxis.set_ticklabels(classes_values, fontsize=15)
ax.yaxis.set_ticklabels(classes_values, fontsize=15)
# plt.savefig(f'images/tflite_confusion_matrix_{data_type}.png', dpi=600)
plt.show()
