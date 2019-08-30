# Keras model to tflite for mobile devices
It is a great time now to develop Artificial Intelligence
apps. There are presence of cools tools to use, from prototyping
to production.

Keras provide easy prototyping and building of models while Tensorflow
provides an easy road map to production. The advantage is that this two
libraries marry/ coexist with each other.

The coexistence provides conversion of keras high level models to tensorflow
low level models e.g to use the models in android there is need to convert them
to tensorflow lite.

This repo shows the exact method to convert keras model to tensorflow lite.

# Examples
We will use a toy data set to show this

import the requred libraries
```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import gfile
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import backend as K
from mute_tf_warnings import tf_mute_warning # this is just a libray to mute tf warnings
```
mute warnings
```python
tf_mute_warning()
```
model directories
```python
save_dir = "models/"
keras_model_name = "keras_model.h5"
tf_lite_model = "tf_lite.tflite"
```
Build a random data set
```python
x = np.vstack((np.random.rand(1000,10), -np.random.rand(1000,10)))
y = np.vstack((np.ones((1000,1)), np.zeros((1000,1))))
```
build a sequential model
```python
model=Sequential()
model.add(Dense(units=64, input_shape=(10,), activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["binary_accuracy"])
```
fit the model
```python
model.fit(x, y, epochs=2, validation_split=0.2)
```
check of the directory exists if not create
```python
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
```
Save keras model to dir
```python
model.save(save_dir+keras_model_name)
```
load the keras model
```python
keras_model = tf.keras.models.load_model(save_dir+keras_model_name)
```
Convert the model to tflite
```python
converter = tf.lite.TFLiteConverter.from_keras_model_file(save_dir+keras_model_name)
tflite_model = converter.convert()

```
Write the tflite model to disk
```python
open(save_dir + tf_lite_model, "wb").write(tflite_model)
```
Load a tflite model and allocate tensors
```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```
Get input and output tensors
```python
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```
Test the Tensorflow Lite model on random data
```python
input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
```
Get tensorflow results
```python
tflite_results = interpreter.get_tensor(output_details[0]["index"])
tflite_results
```
Test the tensorflow keras model on random input_data
```python
tf_results = keras_model.predict(input_data)
```
Compare the results
```python
for tf_result, tflite_result in zip(tf_results, tflite_results):
    try:
        np.testing.assert_almost_equal(tf_result, tflite_result)

    except AssertionError as e:
        print(e)
```

Get the full code [here](https://github.com/kongkip/keras-2-tensorflowlite/blob/master/keras_2_tflite.py)
