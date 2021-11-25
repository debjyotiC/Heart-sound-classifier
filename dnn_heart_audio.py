import tensorflow as tf
import numpy as np
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = np.load('data/mfcc-heart.npz', allow_pickle=True)  # load audio data
x_data, y_data = data['out_x'], data['out_y']  # load into np arrays

seed = 1000
# split data into Train, Validation and Test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=seed, shuffle=True)

wake_word_index = 3  # for murmur
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(52, (2, 2), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(52, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['acc'])
history = model.fit(x_train, y_train, epochs=100, batch_size=20, validation_data=(x_test, y_test))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)
# plot loss
axs[0].plot(epochs, loss, 'bo', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
# plot accuracy
axs[1].plot(epochs, acc, 'bo', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
plt.show()