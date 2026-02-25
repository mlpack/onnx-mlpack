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

#include "operations/gemm.hpp"
#include "operations/relu.hpp"
#include "operations/softmax.hpp"
#include "operations/batch_normalization.hpp"
#include "operations/leaky_relu.hpp"
#include "operations/conv.hpp"
#include "operations/mul.hpp"
#include "operations/add.hpp"
#include "operations/max_pool.hpp"
#include "operations/global_average_pool.hpp"
#include "operations/reshape.hpp"

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
