import onnxruntime as ort
import onnx
import numpy as np
import matplotlib.pyplot as plt

m = onnx.load(open("tinyyolo-v2.3-o8.onnx", "rb"))
onnx.checker.check_model(m)

images = []
for i in range(4):
  # Loads image as shape (H, W, C).
  print(f"loading image yolo_image_{i}.jpg")
  image = plt.imread(f'../test_data/yolo_image_{i}.jpg', format='jpeg')
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
    (4, raw_data.shape[1] * raw_data.shape[2] * raw_data.shape[3]))
print(raw_flattened_data)
np.savetxt("tinyyolo_inputs.csv", raw_flattened_data, delimiter=",")

ort_sess = ort.InferenceSession("tinyyolo-v2.3-o8.onnx")
outputs = []
for i in range(4):
  outputs.append(ort_sess.run(None,
      {'image': raw_data[i:(i + 1), :, :, :]})[0][0])

outputs = np.vstack([x.ravel() for x in outputs])
print(outputs)
print(outputs.shape)

np.savetxt("tinyyolo_outputs.csv", outputs, delimiter=",")
