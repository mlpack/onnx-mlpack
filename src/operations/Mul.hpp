#ifndef MUL_HPP
#define MUL_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddMul(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node);


#include "MUL_impl.hpp"
#endif
