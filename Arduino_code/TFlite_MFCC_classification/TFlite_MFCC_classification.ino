#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "mfcc_model.h"
#include "mfcc_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define DEBUG 0

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
   
  constexpr int kTensorArenaSize = 25*1024; //use 25 for RP2040
  uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(9600);
  
  static tflite::MicroErrorReporter micro_error_reporter;

  error_reporter = &micro_error_reporter;
 
  model = tflite::GetModel(mfcc_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
   
  #if DEBUG
    Serial.println(model_input->dims->size);
    Serial.println(model_input->dims->data[1]);
    Serial.println(model_input->type);

    Serial.println(model_output->dims->size);
    Serial.println(model_output->dims->data[1]);
    Serial.println(model_output->type);
  #endif
}

void loop() {
  #if DEBUG
    unsigned long start_timestamp = micros();
  #endif

  for(int i = 0; i<3900; i++){
    model_input->data.f[i] = mfcc_data_murmur[i];
  }
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  
  float pred_class_1 = model_output->data.f[0];
  float pred_class_2 = model_output->data.f[1];
  
  String result = (pred_class_1 > pred_class_2) ? "murnur" : "normal";

  Serial.print("Predicted Class: ");
  Serial.println(result);
  
#if DEBUG
  Serial.print("Time for inference (us): ");
  Serial.println(micros() - start_timestamp);
#endif

}
