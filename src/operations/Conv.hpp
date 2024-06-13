#ifndef CONV_HPP
#define CONV_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"
#include "../model_parser/helper.hpp"


using namespace std;

void AddConv(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);
              
int FindConvMap(mlpack::FFN<> &ffn, onnx::GraphProto graph, onnx::NodeProto node);

#include "Conv_impl.hpp"
#endif
