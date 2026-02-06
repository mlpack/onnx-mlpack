#ifndef MUL_HPP
#define MUL_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"
#include "../model_parser/helper.hpp"

using namespace std;

vector<size_t> AddMul(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
            onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

class ScaleLayer : public mlpack::Identity<arma::mat>
{
public:
    float scalar;
    ScaleLayer(float scalar) : scalar(scalar) {}

    void Forward(
        const arma::mat &input, arma::mat &output)
    {
        output = input * scalar;
    }
};

float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node);

#include "Mul_impl.hpp"
#endif
