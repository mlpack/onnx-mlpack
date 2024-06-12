#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddLeakyRelu(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);


#include "LeakyRelu_impl.hpp"
#endif
