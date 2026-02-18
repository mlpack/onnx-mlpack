#ifndef GEMM_HPP
#define GEMM_HPP

#include <mlpack.hpp>
#include <onnx/onnx_pb.h>
#include "../onnx_mlpack/utils.hpp"


using namespace std;

inline vector<size_t> AddGemm(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

inline void TransferWeightToGemm(mlpack::DAGNetwork<> &dag, 
    vector<size_t> &layerIndex, 
    onnx::GraphProto &graph, 
    const onnx::NodeProto &node, 
    std::map<std::string, double> onnxOperatorAttribute);

inline size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node);

inline arma::mat ExtractWeights(onnx::GraphProto graph, onnx::NodeProto node, bool transposed);

inline arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node);


#include "Gemm_impl.hpp"
#endif
