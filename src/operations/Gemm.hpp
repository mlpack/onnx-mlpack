#ifndef GEMM_HPP
#define GEMM_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddGemm(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node);

arma::mat ExtractWeights(onnx::GraphProto graph, onnx::NodeProto node, bool transposed);

arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node);


#include "Gemm_impl.hpp"
#endif
