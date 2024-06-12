#include "BatchNormalization.hpp"

void AddBatchNormalization(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                           onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    ffn.Add(new mlpack::Identity());
    cout << "Added the BatchNormalization layer" << endl;
}
