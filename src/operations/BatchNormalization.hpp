#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"
#include "../model_parser/helper.hpp"

using namespace std;

void AddBatchNormalization(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters);


#include "BatchNormalization_impl.hpp"
#endif
