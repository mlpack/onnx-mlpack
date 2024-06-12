#include "MaxPool.hpp"

void AddMaxPool(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    ffn.Add(new mlpack::Identity());
    cout << "Added the MaxPool layer" << endl;
}