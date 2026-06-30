import onnxruntime as ort
import onnx
import numpy as np
import matplotlib.pyplot as plt

m = onnx.load(open("mobilenetv2-7.onnx", "rb"))
onnx.checker.check_model(m)

images = []
for i in range(3):
  # Loads image as shape (H, W, C).
  print(f"loading image imagenet_scaled_{i}.jpg")
  image = plt.imread(f'../test_data/imagenet_scaled_{i}.jpg', format='jpeg')
  print(image.shape)
  images.append(image)

raw_data = np.stack(images, axis=0)
raw_data = np.transpose(raw_data, (0, 3, 1, 2))
raw_data = raw_data.astype(np.float32)
raw_data = raw_data / 255.0
print(raw_data.shape)
print(raw_data)

# Save a copy for loading into mlpack.
raw_flattened_data = np.reshape(raw_data,
    (3, raw_data.shape[1] * raw_data.shape[2] * raw_data.shape[3]))
print(raw_flattened_data)
np.savetxt("mobilenet_inputs.csv", raw_flattened_data, delimiter=",")

ort_sess = ort.InferenceSession("mobilenetv2-7.onnx")
outputs = []
for i in range(3):
  outputs.append(ort_sess.run(None,
      {'data': raw_data[i:(i + 1), :, :, :]})[0][0])

outputs = np.vstack([x.ravel() for x in outputs])
print(outputs)
print(outputs.shape)

np.savetxt("mobilenet_outputs.csv", outputs, delimiter=",")
