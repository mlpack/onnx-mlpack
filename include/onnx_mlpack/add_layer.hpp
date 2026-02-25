/**
 * @file add_layer.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Entrypoint for conversion functions for different ONNX operation types.
 */
#ifndef ONNX_MLPACK_ADD_LAYER_HPP
#define ONNX_MLPACK_ADD_LAYER_HPP

#include "utils.hpp"

#include "operators/gemm.hpp"
#include "operators/relu.hpp"
#include "operators/softmax.hpp"
#include "operators/batch_normalization.hpp"
#include "operators/leaky_relu.hpp"
#include "operators/conv.hpp"
#include "operators/mul.hpp"
#include "operators/add.hpp"
#include "operators/max_pool.hpp"
#include "operators/global_average_pool.hpp"
#include "operators/reshape.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddLayer(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxoperatorAttribute);

inline void TransferWeights(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

} // namespace onnx_mlpack

#include "add_layer_impl.hpp"

#endif
