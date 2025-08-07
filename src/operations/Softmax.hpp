#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "mlpack.hpp"
#include "dag_network.hpp"
#include "onnx_pb.h"

using namespace std;

vector<size_t> AddSoftmax(mlpack::DAGNetwork<> &dag);


#include "Softmax_impl.hpp"
#endif
