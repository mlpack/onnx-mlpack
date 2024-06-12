#ifndef CONV_HPP
#define CONV_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddConv(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);
              
int FindConvMap(onnx::GraphProto graph, onnx::NodeProto node);

#include "Conv_impl.hpp"
#endif
