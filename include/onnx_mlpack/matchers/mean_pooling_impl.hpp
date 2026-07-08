/**
 * @file mean_pooling_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the MaxPooling layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_MEAN_POOLING_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MEAN_POOLING_IMPL_HPP

#include "mean_pooling.hpp"

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
  std::vector<size_t> inputDims;
  ExtractTensorDims(graph, meanPool.input(0), inputDims);
  if (inputDims.size() != 4)
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
  std::vector<size_t> inputDims;
  ExtractTensorDims(graph, meanPool.input(0), inputDims);
  if (inputDims.size() != 4)
  {
    throw std::runtime_error("MeanPoolingSubgraph::Convert(): could not "
        "extract size of input tensor to GlobalAveragePool ONNX operator!");
  }

  network.Add<mlpack::MeanPooling>(inputDims[3] /* width */,
                                   inputDims[2] /* height */);
}

} // namespace onnx_mlpack

#endif
