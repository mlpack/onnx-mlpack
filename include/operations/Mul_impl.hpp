#include "Mul.hpp"

vector<size_t> AddMul(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute){
    float scalar = FindScallingFactor(graph, node);

    size_t a = dag.Add<ScaleLayer>(scalar);
    cout<<"Added ScalarMul layer"<<endl;
    return {a};
}

float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node){
    string initializerName = node.input(1);
    onnx::TensorProto initializer = get::Initializer(graph, initializerName);
    return initializer.float_data(0);
}