#ifndef ADD_HPP
#define ADD_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddAdd(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters);


#include "Add_impl.hpp"
#endif
