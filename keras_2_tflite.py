# import the requred libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import gfile
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import backend as K
from mute_tf_warnings import tf_mute_warning # this is just a libray to mute tf warnings

#mute warnings
tf_mute_warning()

#model directories
save_dir = "models/"
keras_model_name = "keras_model.h5"
tf_lite_model = "tf_lite.tflite"

# Build a random data set
x = np.vstack((np.random.rand(1000,10), -np.random.rand(1000,10)))
y = np.vstack((np.ones((1000,1)), np.zeros((1000,1))))

# build a sequential model
model=Sequential()
model.add(Dense(units=64, input_shape=(10,), activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["binary_accuracy"])

# fit the model
model.fit(x, y, epochs=2, validation_split=0.2)

# check if the directory exists if not create
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save keras model
model.save(save_dir+keras_model_name)

# load the keras model
keras_model = tf.keras.models.load_model(save_dir+keras_model_name)

# Convet the nodel to tflite
converter = tf.lite.TFLiteConverter.from_keras_model_file(save_dir+keras_model_name)
tflite_model = converter.convert()

# Write the tflite model to disk
open(save_dir+tf_lite_model, "wb").write(tflite_model)

# Load a tflite model and allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the Tensorflow Lite model on ransom data
input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Get tensorflow results
tflite_results = interpreter.get_tensor(output_details[0]["index"])

# Test the tensorflow keras model on random input_data
tf_results = keras_model.predict(input_data)

# Compare the results
for tf_result, tflite_result in zip(tf_results, tflite_results):
    try:
        np.testing.assert_almost_equal(tf_result, tflite_result)
        print("Keras and tensorflow lite results are equal")

    except AssertionError as e:
        print(e)
