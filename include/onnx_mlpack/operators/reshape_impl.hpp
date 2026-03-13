/**
 * @file reshape_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX Reshape operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_RESHAPE_IMPL_HPP
#define ONNX_MLPACK_OPERATORS_RESHAPE_IMPL_HPP

#include "reshape.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddReshape(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  std::vector<int> requiredDimensions = FindReshapedDimension(graph, node);
  // vector<size_t> outputDimension(requireDimensions.begin() + 0, requireDimensions.begin() + 3);
  std::vector<size_t> outputDimension(3, 1);
  // just to get the dimension of the incoming i will be reseting the ffn
  mlpack::DAGNetwork<> dag_ = dag;
  dag_.Reset();
  std::vector<size_t> inputDimension =
      dag_.Network().back()->OutputDimensions();

  for (int i = 0; i < 3; i++)
  {
    if (requiredDimensions[i] == 0)
      outputDimension[i] = inputDimension[i];
    else if (requiredDimensions[i] > 0)
      outputDimension[i] = requiredDimensions[i];
  }
  for (int i = 0; i < 3; i++)
  {
    if (requiredDimensions[i] == -1)
    {
      std::vector<size_t> rough = outputDimension;
      rough[i] = 1;
      outputDimension[i] = std::accumulate(inputDimension.begin(),
          inputDimension.end(), 1, std::multiplies<int>()) /
          std::accumulate(rough.begin(), rough.end(), 1,
          std::multiplies<int>());
    }
  }

  size_t a = dag.Add<Reshape>(outputDimension);
  std::cout << "Added mlpack::Reshape Layer" << std::endl;
  return {a};
}

inline std::vector<int> FindReshapedDimension(
    onnx::GraphProto graph,
    onnx::NodeProto node)
{
  // from this initializer we will be getting the dimension of the output of
  // layer
  std::string initializerName = node.input(1);
  onnx::TensorProto initializer = Initializer(graph, initializerName);

  std::vector<int> dimensions(4, 1); /// colMajor W, H, C, C

  int j = 0;
  for (int i = initializer.dims(0) - 1; i >= 0; i--)
  {
    dimensions[j] = initializer.int64_data(i);
    j++;
  }

  return dimensions;
}

} // namespace onnx_mlpack

#endif
