#!/usr/bin/python3
#
# Generate a 3-layer Linear network with TensorFlow, and export to ONNX.
import numpy as np
import tensorflow as tf
import tf2onnx
import subprocess

init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, use_bias=False, kernel_initializer=init,
            input_shape=(100,)),
        tf.keras.layers.Dense(10, use_bias=False, kernel_initializer=init),
        tf.keras.layers.Dense(3, use_bias=False, kernel_initializer=init)])

inputs = np.random.randn(3, 100).astype(np.float32)
outputs = model.predict(inputs)

np.savetxt("tf_linear_no_bias_inputs.csv", inputs, delimiter=',')
np.savetxt("tf_linear_no_bias_outputs.csv", outputs, delimiter=',')

model.export("tf_linear_no_bias")
# This is seriously what their documentation suggests in the examples.
proc = subprocess.run('python -m tf2onnx.convert --saved-model tf_linear_no_bias --output tf_linear_no_bias.onnx --opset 12'.split(), capture_output=True)

print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))
