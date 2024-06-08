#include "Mul.hpp"

void AddMul(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node){
    float scalar = FindScallingFactor(graph, node);
    ffn.Add(new ScaleLayer(scalar));
    cout<<"added the Mul"<<endl;
}

class ScaleLayer: public mlpack::IdentityType<arma::mat>{
public:
    float scalar;
    ScaleLayer(fload scalar) : scalar(scalar){}

    void Forward(
        const arma::mat& input, arma::mat& output)
    {
    output = input * scalar;
    }
}

float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node){
    string initializerName = node.input(1);
    onnx::TensorProt initializer = get::Initializer(initializerName);
    return initializer.float_data(0);
}
