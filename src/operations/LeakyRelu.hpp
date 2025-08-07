#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "mlpack.hpp"
#include "dag_network.hpp"
#include "onnx_pb.h"

using namespace std;

vector<size_t> AddLeakyRelu(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);


#include "LeakyRelu_impl.hpp"
#endif
