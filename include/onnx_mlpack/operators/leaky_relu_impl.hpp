/**
 * @file leaky_relu_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX LeakyReLU operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_LEAKY_RELU_IMPL_HPP
#define ONNX_MLPACK_OPERATORS_LEAKY_RELU_IMPL_HPP

#include "leaky_relu.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddLeakyRelu(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  double alpha = onnxOperatorAttribute["alpha"];
  size_t a = dag.Add<mlpack::LeakyReLU>(alpha);
  std::cout << "Added mlpack::LeakyRelu Layer" << std::endl;
  return {a};
}

} // namespace onnx_mlpack

#endif
