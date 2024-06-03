#ifndef ADD_LAYER_HPP
#define ADD_LAYER_HPP

#include "gemm.hpp"
#include "relu.hpp"
#include "softmax.hpp"

void AddLayer(mlpack::FFN<> &ffn,
              onnx::GraphProto graph,
              onnx::NodeProto node,
              std::map<string, double> onnxoperatorAttribute);


#include "add_layer_impl.hpp"
#endif