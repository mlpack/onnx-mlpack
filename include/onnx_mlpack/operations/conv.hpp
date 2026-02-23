/**
 * @file conv.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX Conv operation conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_CONV_HPP
#define ONNX_MLPACK_OPERATIONS_CONV_HPP

#include "../helper.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddConv(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

inline void TransferWeightToConv(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

inline int FindConvMap(mlpack::DAGNetwork<>& dag,
                       onnx::GraphProto graph,
                       onnx::NodeProto node);

} // namespace onnx_mlpack

#include "conv_impl.hpp"

#endif
