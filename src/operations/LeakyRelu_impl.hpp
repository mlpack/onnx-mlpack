#include "LeakyRelu.hpp"

void AddLeakyRelu(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                  onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    ffn.Add(new mlpack::Identity());
    cout << "Added the LeakyRelu layer" << endl;
}