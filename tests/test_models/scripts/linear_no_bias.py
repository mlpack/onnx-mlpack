# linear_no_bias.py: manually build ONNX graph for three-layer LinearNoBias
# network.
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

input_size = 12
layer_size = 10
num_layers = 3

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT,
    [None, input_size])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT,
    [None, layer_size])

nodes = []
initializers = []
prev = "input"

for i in range(num_layers):
    w_name = f"W{i+1}"
    out_name = "output" if i == num_layers - 1 else f"h{i+1}"

    if i == 0:
        shape = (input_size, layer_size)
    else:
        shape = (layer_size, layer_size)

    # Use random weights.
    W = np.random.randn(*shape).astype(np.float32)

    weight_tensor = helper.make_tensor(name=w_name, data_type=TensorProto.FLOAT,
        dims=shape, vals=W.flatten())

    initializers.append(weight_tensor)

    node = helper.make_node("MatMul", inputs=[prev, w_name], outputs=[out_name],
        name=f"layer{i+1}")

    nodes.append(node)
    prev = out_name

# Put nodes together into a graph.
graph = helper.make_graph(nodes, "three_layer_net", [input_tensor],
    [output_tensor], initializer=initializers)

model = helper.make_model(graph, producer_name="example")
onnx.save(model, "three_layer_net.onnx")

# Now run a couple of points through it.
session = ort.InferenceSession(model.SerializeToString())

# Get input/output names.
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
x = np.random.randn(3, 12).astype(np.float32)
y = session.run([output_name], {input_name: x})[0]

print(x)
print(y)
