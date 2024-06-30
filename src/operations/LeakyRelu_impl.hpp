#include "LeakyRelu.hpp"

void AddLeakyRelu(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                  onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters)
{
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(new mlpack::Identity());
    cout << "Added the LeakyRelu layer" << endl;
}