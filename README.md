
# ONNX-mlpack Translator

<div align="center">
  <a href="http://mlpack.org">
    <img src="img/onnx-mlpack.png" alt="ONNX-mlpack Translator" height="70%" width="70%">
  </a>
  <br>
  <h2>Unlock the Power of Other Frameworks in mlpack</h2>
</div>

### Dependencies of this repository
* Protocol Buffer
* ONNX

ONNX internally download and build protobuf for ONNX build. So you just need to build ONNX. For building onnx follow their [ReadMe](ONNX will internally download and build protobuf for ONNX build.).

### The repository is in developing phase and its been tested on the following models.

| Models    | Model Generation   | Weight Transfer |     |
| --------- | ------------------ | --------------- | --- |
| Linear NN | :heavy_check_mark: | :x:             |     |
| Mnist     | :heavy_check_mark: | :x:             |     |
