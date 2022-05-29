import tensorflow as tf
import numpy as np

model_type = "mfe"
type_of_quantization = "Default"
saved_model_dir = f"saved_model/{model_type}"
BATCH_SIZE = 65

data = np.load("data/lmfe.npz", allow_pickle=True)
x_data, _ = data['out_x'].astype(np.float32), data['out_y'].astype(np.float32)


def representative_dataset():
    mfcc = tf.data.Dataset.from_tensor_slices(x_data).batch(1)
    for i in mfcc.take(BATCH_SIZE):
        yield [i]


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
