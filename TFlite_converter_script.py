import tensorflow as tf
import numpy as np

model_type = "mfcc"
type_of_quantization = "int8"
saved_model_dir = f"saved_model/{model_type}"


def representative_dataset():
    for _ in range(100):
        data = np.random.rand(283, 3900)
        yield [data.astype(np.float32)]


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
if type_of_quantization == "Default":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model.
    with open(f'saved_tflite_model/{model_type}_default.tflite', 'wb') as f:
        f.write(tflite_model)
elif type_of_quantization == "int8":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open(f'saved_tflite_model/{model_type}_int8.tflite', 'wb') as f:
        f.write(tflite_model)
elif type_of_quantization == "float16":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # Save the model.
    with open(f'saved_tflite_model/{model_type}_float16.tflite', 'wb') as f:
        f.write(tflite_model)
