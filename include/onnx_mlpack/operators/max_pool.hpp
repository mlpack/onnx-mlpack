/**
 * @file max_pool.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX MaxPool operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_MAX_POOL_HPP
#define ONNX_MLPACK_OPERATORS_MAX_POOL_HPP

#include "../utils.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddMaxPool(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

} // namespace onnx_mlpack

#include "max_pool_impl.hpp"

#endif
