import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split

data = np.load("data/mfcc-murmur-normal.npz", allow_pickle=True)
x_data, y_data = data['out_x'], data['out_y']

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=1, shuffle=True)
# print(x_test[0].shape)
# loaded_model = tf.keras.models.load_model('saved_model/mfcc')
# loaded_model.summary()
# # Evaluate the restored model
# loss, acc = loaded_model.evaluate(x_test, y_test, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# load Keras model
load_model = tf.keras.models.load_model('saved_model/mfcc')
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_test.shape)
classes = load_model.predict(x_test[1:])


print(classes)

