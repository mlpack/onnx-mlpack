#ifndef ADD_LAYER_HPP
#define ADD_LAYER_HPP

#include "Gemm.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "BatchNormalization.hpp"
#include "LeakyRelu.hpp"
#include "Conv.hpp"
#include "Mul.hpp"
#include "Add.hpp"
#include "MaxPool.hpp"
#include "GlobalAveragePool.hpp"
#include "Reshape.hpp"

#include "../model_parser/utils.hpp"

/**
 * @brief 
 */
vector<size_t> AddLayer(mlpack::DAGNetwork<> &dag,
              onnx::GraphProto &graph,
              const onnx::NodeProto &node,
              std::map<string, double> onnxoperatorAttribute);

void TransferWeights(mlpack::DAGNetwork<> &dag, 
    vector<size_t> &layerIndex, 
    onnx::GraphProto &graph, 
    const onnx::NodeProto &node, 
    std::map<std::string, double> onnxOperatorAttribute);


#include "add_layer_impl.hpp"
#endif