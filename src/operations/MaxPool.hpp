#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "mlpack.hpp"
#include "dag_network.hpp"
#include "onnx_pb.h"
#include "../model_parser/utils.hpp"

using namespace std;

vector<size_t> AddMaxPool(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);


#include "MaxPool_impl.hpp"
#endif
