#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"
#include "../model_parser/utils.hpp"

using namespace std;

void AddMaxPool(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters);


#include "MaxPool_impl.hpp"
#endif
