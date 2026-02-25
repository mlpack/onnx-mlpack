/**
 * @file softmax.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX Softmax operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_SOFTMAX_HPP
#define ONNX_MLPACK_OPERATORS_SOFTMAX_HPP

namespace onnx_mlpack {

inline std::vector<size_t> AddSoftmax(mlpack::DAGNetwork<> &dag);

} // namespace onnx_mlpack

#include "softmax_impl.hpp"

#endif
