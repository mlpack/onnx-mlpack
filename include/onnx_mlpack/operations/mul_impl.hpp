/**
 * @file mul_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of ONNX Mul operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_MUL_IMPL_HPP
#define ONNX_MLPACK_OPERATIONS_MUL_IMPL_HPP

#include "mul.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddMul(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  float scalar = FindScallingFactor(graph, node);

  size_t a = dag.Add<ScaleLayer>(scalar);
  std::cout << "Added ScalarMul layer" << std::endl;
  return {a};
}

inline float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node)
{
  std::string initializerName = node.input(1);
  onnx::TensorProto initializer = Initializer(graph, initializerName);
  return initializer.float_data(0);
}

} // namespace onnx_mlpack

#endif
