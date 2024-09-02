# Evaluation script to compare full and lite models

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Load the full model
full_model = tf.keras.models.load_model('../models/model_full.h5')
full_model_loss, full_model_acc = full_model.evaluate(x_test, y_test, verbose=2)

# Load the optimized model
interpreter = tf.lite.Interpreter(model_path='../models/model_optimized.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate the optimized model
correct_predictions = 0
for i in range(len(x_test)):
    input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == y_test[i]:
        correct_predictions += 1

lite_model_acc = correct_predictions / len(x_test)

print(f"Full model accuracy: {full_model_acc}")
print(f"Lite model accuracy: {lite_model_acc}")
