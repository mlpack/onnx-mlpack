/**
 * @file global_average_pool.hpp
 * @author Kumar Utkarsh
 *
 * Handling of GlobalAveragePool ONNX operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_GLOBAL_AVERAGE_POOL_HPP
#define ONNX_MLPACK_OPERATORS_GLOBAL_AVERAGE_POOL_HPP

namespace onnx_mlpack {

inline std::vector<size_t> AddGlobalAveragePool(mlpack::DAGNetwork<>& dag);

} // namespace onnx_mlpack

#include "global_average_pool_impl.hpp"

#endif
