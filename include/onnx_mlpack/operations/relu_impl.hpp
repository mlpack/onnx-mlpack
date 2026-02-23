/**
 * @file relu_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX Relu operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_RELU_IMPL_HPP
#define ONNX_MLPACK_OPERATIONS_RELU_IMPL_HPP

#include "relu.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddRelu(mlpack::DAGNetwork<>& dag)
{
  size_t a = dag.Add<mlpack::LeakyReLU>(0);
  std::cout << "Added mlpack::Relu Layer" << std::endl;
  return {a};
}

} // namespace onnx_mlpack

#endif
