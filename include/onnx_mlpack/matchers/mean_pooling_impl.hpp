/**
 * @file mean_pooling_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the MaxPooling layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MEAN_POOLING_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MEAN_POOLING_IMPL_HPP

#include "mean_pooling.hpp"
#include "../extract_attribute.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a MeanPooling layer.
 */
inline bool MeanPoolingSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MaxPool to ensure that we actually can
  // do the conversion.
  const onnx::NodeProto& meanPool = graph.node(nodes[0]);
  if (meanPool.op_type() != "GlobalAveragePool")
    return false;

  // The ONNX GlobalAveragePool operator is really quite simple.  It requires a
  // four-dimensional input tensor of shape (N x C x H x W) where N is the batch
  // size, and pools across channels to produce an output of shape (N x C).

  // The only thing we need to check before conversion is that we can extract
  // the shape of the input tensor.
  const std::string& inputName = meanPool.input(0);
  size_t inputHeight = 0;
  size_t inputWidth = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == inputName && t.dims_size() == 4)
    {
      inputHeight = t.dims(2);
      inputWidth = t.dims(3);
      break;
    }
  }

  if (inputHeight == 0 || inputWidth == 0)
  {
    for (size_t i = 0; i < graph.value_info_size(); ++i)
    {
      const onnx::ValueInfoProto& v = graph.value_info(i);
      if (v.has_name() && v.name() == inputName && v.has_type() &&
          v.type().has_tensor_type() &&
          v.type().tensor_type().has_shape() &&
          v.type().tensor_type().shape().dim_size() == 4 &&
          v.type().tensor_type().shape().dim(2).has_dim_value() &&
          v.type().tensor_type().shape().dim(3).has_dim_value())
      {
        inputHeight = v.type().tensor_type().shape().dim(2).dim_value();
        inputWidth = v.type().tensor_type().shape().dim(3).dim_value();
      }
    }
  }

  if (inputHeight == 0 || inputWidth == 0)
    return false;

  return true;
}

/**
 * Create a MeanPooling layer with the same metadata as the given ONNX graph.
 */
inline void MeanPoolingSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& meanPool = graph.node(nodes[0]);

  // In order to convert to a MeanPooling layer, we need to know the size of the
  // input tensor.  The size of the kernel we will use is the full size of the
  // tensor's width and height.
  const std::string& inputName = meanPool.input(0);
  size_t inputHeight = 0;
  size_t inputWidth = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == inputName && t.dims_size() == 4)
    {
      inputHeight = t.dims(2);
      inputWidth = t.dims(3);
      break;
    }
  }

  if (inputHeight == 0 || inputWidth == 0)
  {
    for (size_t i = 0; i < graph.value_info_size(); ++i)
    {
      const onnx::ValueInfoProto& v = graph.value_info(i);
      if (v.has_name() && v.name() == inputName && v.has_type() &&
          v.type().has_tensor_type() &&
          v.type().tensor_type().has_shape() &&
          v.type().tensor_type().shape().dim_size() == 4 &&
          v.type().tensor_type().shape().dim(2).has_dim_value() &&
          v.type().tensor_type().shape().dim(3).has_dim_value())
      {
        inputHeight = v.type().tensor_type().shape().dim(2).dim_value();
        inputWidth = v.type().tensor_type().shape().dim(3).dim_value();
      }
    }
  }

  if (inputHeight == 0 || inputWidth == 0)
  {
    throw std::runtime_error("MeanPoolingSubgraph::Convert(): could not "
        "extract size of input tensor to GlobalAveragePool ONNX operator!");
  }

  network.Add<mlpack::MeanPooling>(inputWidth, inputHeight);
}

} // namespace onnx_mlpack

#endif
