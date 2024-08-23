#include "LeakyRelu.hpp"

void AddLeakyRelu(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                  onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters)
{
    double alpha = onnxOperatorAttribute["alpha"];
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(new mlpack::LeakyReLU(alpha));
    cout << "Added mlpack::LeakyRelu Layer" << endl;
}