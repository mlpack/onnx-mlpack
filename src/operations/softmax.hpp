#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddSoftmax(mlpack::FFN<> &ffn);


#include "softmax_impl.hpp"
#endif
