# script with full model optimization in: model_full out: model_optimized

import tensorflow as tf

# Load the full model
model = tf.keras.models.load_model('../models/model_full.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the optimized model
with open('../models/model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)