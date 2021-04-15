import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model/fft-clf')

tflite_model = converter.convert()

open("model/tflite_model/converted_model.tflite", "wb").write(tflite_model)
