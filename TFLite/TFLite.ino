
#include <EloquentTinyML.h>
#include "c_to_f_model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;


void setup() {
  Serial.begin(115200);
  ml.begin(c_to_f_model);
}

void loop() {
  float x = 100;
  float input[1] = { x };
  float predicted = ml.predict(input);
  Serial.print(x);
  Serial.print(" degree C is: ");
  Serial.print(predicted);
  Serial.println(" F");    
  delay(1000);
}
