#ifndef CONV_HPP
#define CONV_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"
#include "../model_parser/helper.hpp"


using namespace std;

inline vector<size_t> AddConv(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

inline void TransferWeightToConv(mlpack::DAGNetwork<> &dag,
                          vector<size_t> &layerIndex,
                          onnx::GraphProto &graph,
                          const onnx::NodeProto &node,
                          std::map<std::string, double> onnxOperatorAttribute);

inline int FindConvMap(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph, onnx::NodeProto node);

#include "Conv_impl.hpp"
#endif
