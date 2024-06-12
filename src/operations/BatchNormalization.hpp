#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddBatchNormalization(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);


#include "BatchNormalization_impl.hpp"
#endif
