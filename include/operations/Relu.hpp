#ifndef RELU_HPP
#define RELU_HPP

#include "mlpack.hpp"
#include <onnx/onnx_pb.h>

using namespace std;

inline vector<size_t> AddRelu(mlpack::DAGNetwork<> &dag);


#include "Relu_impl.hpp"
#endif
