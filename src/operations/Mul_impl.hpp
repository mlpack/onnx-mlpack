#include "Mul.hpp"

void AddMul(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute){
    float scalar = FindScallingFactor(graph, node);
    ffn.Add(new ScaleLayer(scalar));
    cout<<"Added the Mul layer"<<endl;
}

float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node){
    string initializerName = node.input(1);
    onnx::TensorProto initializer = get::Initializer(graph, initializerName);
    return initializer.float_data(0);
}
