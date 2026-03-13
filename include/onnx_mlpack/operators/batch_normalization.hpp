/**
 * @file batch_normalization.hpp
 * @author Kumar Utkarsh
 *
 * Handling for ONNX BatchNormalization operation.
 */
#ifndef ONNX_MLPACK_OPERATORS_BATCH_NORMALIZATION_HPP
#define ONNX_MLPACK_OPERATORS_BATCH_NORMALIZATION_HPP

#include "../helper.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddBatchNormalization(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

inline void TransferWeightToBatchNormalization(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

} // namespace onnx_mlpack

#include "batch_normalization_impl.hpp"

#endif
