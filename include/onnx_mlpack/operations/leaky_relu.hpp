/**
 * @file leaky_relu.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX LeakyReLU operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_LEAKY_RELU_HPP
#define ONNX_MLPACK_OPERATIONS_LEAKY_RELU_HPP

namespace onnx_mlpack {

inline std::vector<size_t> AddLeakyRelu(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

} // namespace onnx_mlpack

#include "leaky_relu_impl.hpp"

#endif
