#include "LeakyRelu.hpp"

inline vector<size_t> AddLeakyRelu(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
                  onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    double alpha = onnxOperatorAttribute["alpha"];
    // layerParameters.push_back(arma::Mat<double>());
    size_t a = dag.Add<mlpack::LeakyReLU>(alpha);
    cout << "Added mlpack::LeakyRelu Layer" << endl;
    return {a};
}
