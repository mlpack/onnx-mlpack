#!/usr/bin/python3
#
# Generate a network with exact GELU activations that can be mapped to mlpack.
# This one is done specifically because tf2onnx has a bug that emits an invalid
# ONNX operation name.  Our tool is fine with that, but it breaks shape
# inference, so it breaks matching for other layers.

import numpy as np
import tensorflow as tf
import tf2onnx
import subprocess

init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, kernel_initializer=init, bias_initializer=init,
        input_shape=(100,)),
    tf.keras.layers.Activation(tf.keras.activations.gelu),
    tf.keras.layers.Dense(25, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.gelu),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.gelu),
    tf.keras.layers.Dense(3, kernel_initializer=init, bias_initializer=init)])

inputs = np.random.rand(3, 100).astype(np.float32)
outputs = model.predict(inputs)

np.savetxt("tf_gelu_activation_inputs.csv", inputs, delimiter=',')
np.savetxt("tf_gelu_activation_outputs.csv", outputs, delimiter=',')

model.export("tf_gelu_activation")
# This is seriously what their documentation suggests in the examples.
proc = subprocess.run('python -m tf2onnx.convert --saved-model tf_gelu_activation --output tf_gelu_activation.onnx --opset 12'.split(), capture_output=True)

print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))
