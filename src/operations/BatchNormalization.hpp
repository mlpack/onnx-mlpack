#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "mlpack.hpp"
#include "dag_network.hpp"
#include "onnx_pb.h"
#include "../model_parser/helper.hpp"

using namespace std;

vector<size_t> AddBatchNormalization(mlpack::DAGNetwork<> &dag, onnx::GraphProto &graph,
              const onnx::NodeProto &node, map<string, double> onnxOperatorAttribute);

void TransferWeightToBatchNormalization(mlpack::DAGNetwork<> &dag,
                                        vector<size_t> &layerIndex,
                                        onnx::GraphProto &graph,
                                        const onnx::NodeProto &node,
                                        std::map<std::string, double> onnxOperatorAttribute);

#include "BatchNormalization_impl.hpp"
#endif
