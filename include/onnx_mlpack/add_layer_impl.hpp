/**
 * @file add_layer_impl.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Entrypoint for conversion from ONNX operations to mlpack layers.
 */
#ifndef ONNX_MLPACK_ADD_LAYER_IMPL_HPP
#define ONNX_MLPACK_ADD_LAYER_IMPL_HPP

#include "add_layer.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddLayer(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  if (node.op_type() == "Gemm")
  {
    return AddGemm(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "Softmax")
  {
    return AddSoftmax(dag);
  }
  else if (node.op_type() == "Relu")
  {
    return AddRelu(dag);
  }
  else if (node.op_type() == "LeakyRelu")
  {
    return AddLeakyRelu(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "Mul")
  {
    // scalar multiplication
    return AddMul(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "Add")
  {
    return AddAdd(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "Conv")
  {
    return AddConv(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "BatchNormalization")
  {
    return AddBatchNormalization(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "MaxPool")
  {
    return AddMaxPool(dag, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "GlobalAveragePool")
  {
    AddGlobalAveragePool(dag);
  }
  else if (node.op_type() == "Reshape")
  {
    return AddReshape(dag, graph, node, onnxOperatorAttribute);
  }
  else
  {
    throw std::runtime_error("ONNX operation '" + node.op_type() + "' not " +
        "implemented!");
  }

  return std::vector<size_t>();
}

inline void TransferWeights(mlpack::DAGNetwork<>& dag,
                            std::vector<size_t>& layerIndex,
                            onnx::GraphProto& graph,
                            const onnx::NodeProto& node,
                            std::map<std::string, double> onnxOperatorAttribute)
{
  if (node.op_type() == "Gemm")
  {
    TransferWeightToGemm(dag, layerIndex, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "Softmax")
  {
    return;
  }
  else if (node.op_type() == "Relu")
  {
    return;
  }
  else if (node.op_type() == "LeakyRelu")
  {
    return;
  }
  else if (node.op_type() == "Mul")
  {
    return;
    // TransferWeightToMul(layer, node); // scalar multiplication
  }
  else if (node.op_type() == "Add")
  {
    return;
    // TransferWeightToAdd(layer, node);
  }
  else if (node.op_type() == "Conv")
  {
    TransferWeightToConv(dag, layerIndex, graph, node, onnxOperatorAttribute);
  }
  else if (node.op_type() == "BatchNormalization")
  {
    TransferWeightToBatchNormalization(dag, layerIndex, graph, node,
        onnxOperatorAttribute);
  }
  else if (node.op_type() == "MaxPool")
  {
    return;
    // TransferWeightToMaxPool(layer, node);
  }
  else if (node.op_type() == "GlobalAveragePool")
  {
    return;
    // TransferWeightToGlobalAveragePool(dag);
  }
  else if (node.op_type() == "Reshape")
  {
    return;
  }
  else
  {
    throw std::runtime_error("ONNX operation '" + node.op_type() + "' not " +
        "implemented!");
  }
}

} // namespace onnx_mlpack

#endif
