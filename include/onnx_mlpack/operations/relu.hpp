/**
 * @file relu.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX Relu operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_RELU_HPP
#define ONNX_MLPACK_OPERATIONS_RELU_HPP

namespace onnx_mlpack {

inline std::vector<size_t> AddRelu(mlpack::DAGNetwork<>& dag);

} // namespace onnx_mlpack

#include "relu_impl.hpp"

#endif
