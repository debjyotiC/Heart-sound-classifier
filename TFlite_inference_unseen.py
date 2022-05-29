import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

feature_type = "lmfe"

data = np.load(f"data/{feature_type}.npz", allow_pickle=True)
x_data, y_data = data['out_x'], data['out_y']
classes_values = ["murmur", "normal"]
classes = len(classes_values)

train_ratio = 0.80
validation_ratio = 0.15
test_ratio = 0.05

# train is now 75% of the entire data set
# test is now 10% of the initial data set
# validation is now 15% of the initial data set
y_data = tf.keras.utils.to_categorical(y_data - 1, classes)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio, random_state=0,  stratify=y_data)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0, stratify=y_test)

input_length = x_train[0].shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))


model = tf.keras.Sequential([
    tf.keras.layers.Reshape((150, 26), input_shape=(input_length,)),

    tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),

    # Dense layer
    tf.keras.layers.Dense(classes, activation='softmax', name='y_pred')
])

# model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['acc'])

# this controls the batch size
BATCH_SIZE = 50
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=150, validation_data=validation_dataset)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

prediction = model.predict(x_test)
predicted = np.argmax(prediction, axis=1)
actual = np.argmax(y_test, axis=1)

results = confusion_matrix(actual, predicted)
acc_test = accuracy_score(actual, predicted)


print('Accuracy Score :', acc_test)
print(f'Classification report for {feature_type.upper()} model: ')
print(classification_report(actual, predicted))
print(results)
epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(3, 1)

# plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')
# plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc_test')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc_test')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
sns.heatmap(results, annot=True, ax=axs[2], fmt='g')
axs[2].set_xlabel('Predicted labels')
axs[2].set_ylabel('True labels')
# axs[2].set_title(f'Confusion Matrix for {feature_type.upper()} TFLite model accuracy {round(acc, 2)}')
axs[2].xaxis.set_ticklabels(classes_values)
axs[2].yaxis.set_ticklabels(classes_values)
plt.show()
