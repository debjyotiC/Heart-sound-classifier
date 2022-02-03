import tensorflow as tf

saved_model_dir = "saved_model/mfcc_flattened"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('saved_tflite_model/mfcc_flattened.tflite', 'wb') as f:
    f.write(tflite_model)
