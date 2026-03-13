/**
 * @file softmax_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX Softmax operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_SOFTMAX_IMPL_HPP
#define ONNX_MLPACK_OPERATORS_SOFTMAX_IMPL_HPP

#include "softmax.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddSoftmax(mlpack::DAGNetwork<> &dag)
{
  size_t a = dag.Add<mlpack::Softmax>();
  std::cout << "Added mlpack::Softmax Layer" << std::endl;
  return {a};
}

} // namespace onnx_mlpack

#endif
