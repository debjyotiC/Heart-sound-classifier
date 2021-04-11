import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

df = pd.read_csv("data/fft-data-3.csv").dropna().reset_index(drop=True)

x = df.drop(columns=['Frequency', 'Label']).to_numpy(dtype='float64')
y = df['Label'].to_numpy(dtype='float64')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(200, input_dim=x_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(170, activation='relu'),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(70, activation='softmax'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

model_out = model.fit(x_train, y_train, epochs=150, validation_data=[x_test, y_test], batch_size=20)

model.save("model/fft-clf")

print("Training accuracy: {:.5f}".format(np.mean(model_out.history['accuracy'])))
print("Validation accuracy: {:.5f}".format(np.mean(model_out.history['val_accuracy'])))

y_prediction = model.predict(x_test)
print(f"Actual class is {y_test} and predicted class is {y_prediction[0]}")

fig, axs = plt.subplots(2, 1)
# plot loss
axs[0].plot(model_out.history['loss'], color='Green', label='Train loss')
axs[0].plot(model_out.history['val_loss'], color='Blue', label='Val loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# plot accuracy
axs[1].plot(model_out.history['accuracy'], color='Red', label='Train acc')
axs[1].plot(model_out.history['val_accuracy'], color='Black', label='Val acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
plt.legend()
plt.show()
