import onnxruntime as ort
import onnx
import numpy as np

m = onnx.load(open("onnc-lenet5.onnx", "rb"))
onnx.checker.check_model(m)

raw_data = np.genfromtxt("onnc_lenet5_inputs.csv", delimiter=",")
raw_data = np.reshape(raw_data, [3, 1, 28, 28]).astype(np.float32)

ort_sess = ort.InferenceSession("onnc-lenet5.onnx")
outputs = []
# absurd that I have to do this
for i in range(raw_data.shape[0]):
    outputs.append(ort_sess.run(None,
        {'import/Placeholder:0': raw_data[i:(i + 1), :, :, :]})[0][0])

out_data = np.array(outputs)
out_data = np.reshape(out_data, [3, 10])
print(out_data.shape)
print(out_data)

np.savetxt("onnc_lenet5_outputs.csv", out_data, delimiter=",")
