#!/usr/bin/python3
#
# Generate a network with a whole bunch of different activations that can be
# mapped to mlpack.

import numpy as np
import tensorflow as tf
import tf2onnx
import subprocess

init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, kernel_initializer=init, bias_initializer=init,
        input_shape=(100,)),
    # It's a joke that I have to do this to pass a parameter.
    tf.keras.layers.Activation(
        lambda x : tf.keras.activations.elu(x, alpha=0.98)),
    tf.keras.layers.Dense(25, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(
        lambda x : tf.keras.activations.gelu(x, approximate=True)),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    # mlpack does not support HardSigmoid with a=6 (only a=5).
#    tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid),
#    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.mish),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.softplus),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.tanh),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.relu),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    # leaky ReLU
    tf.keras.layers.ReLU(negative_slope=0.1),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init),
    tf.keras.layers.Activation(tf.keras.activations.selu),
    tf.keras.layers.Dense(10, kernel_initializer=init, bias_initializer=init, name="preprelu"),
    tf.keras.layers.PReLU(alpha_initializer=init, shared_axes=[1], name="prelu"),
    tf.keras.layers.Dense(3, kernel_initializer=init, bias_initializer=init)])

inputs = np.random.rand(3, 100).astype(np.float32)
outputs = model.predict(inputs)

int_model2 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("preprelu").output)
int_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("prelu").output)
print(int_model2.predict(inputs))
print(int_model.predict(inputs))

np.savetxt("tf_activations_inputs.csv", inputs, delimiter=',')
np.savetxt("tf_activations_outputs.csv", outputs, delimiter=',')

model.export("tf_activations")
# This is seriously what their documentation suggests in the examples.
proc = subprocess.run('python -m tf2onnx.convert --saved-model tf_activations --output tf_activations.onnx --opset 12'.split(), capture_output=True)

print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))
