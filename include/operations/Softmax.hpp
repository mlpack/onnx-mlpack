#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

inline vector<size_t> AddSoftmax(mlpack::DAGNetwork<> &dag);


#include "Softmax_impl.hpp"
#endif
