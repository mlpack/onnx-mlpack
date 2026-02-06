#ifndef GEMM_HPP
#define GEMM_HPP

#include <mlpack.hpp>
#include "onnx_pb.h"
#include "../model_parser/utils.hpp"


using namespace std;

vector<size_t> AddGemm(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

void TransferWeightToGemm(mlpack::DAGNetwork<> &dag, 
    vector<size_t> &layerIndex, 
    onnx::GraphProto &graph, 
    const onnx::NodeProto &node, 
    std::map<std::string, double> onnxOperatorAttribute);

size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node);

arma::mat ExtractWeights(onnx::GraphProto graph, onnx::NodeProto node, bool transposed);

arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node);


#include "Gemm_impl.hpp"
#endif
