import os

tflite_model_path = "saved_tflite_model/mfcc_int8.tflite"
tflite_micro_model_path = "saved_tflite_micro_model/mfcc_int8.h"

converter_path = "xxd -i " + tflite_model_path + " > " + tflite_micro_model_path

print(converter_path)

os.system(converter_path)
