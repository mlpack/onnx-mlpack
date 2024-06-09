#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddBatchNormalization(mlpack::FFN<> &ffn);


#include "BatchNormalization_impl.hpp"
#endif
