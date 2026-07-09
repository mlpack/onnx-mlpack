/**
 * @file softmax_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Softplus layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_SOFTMAX_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SOFTMAX_IMPL_HPP

#include "softmax.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Softmax layer.
 */
inline bool SoftmaxSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // There are no parameters to the Softplus layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& softmax = graph.node(nodes[0]);
  if (softmax.op_type() != "Softmax")
    return false;

  // mlpack's Softmax layer operates on *all* inputs, so, to match the ONNX
  // semantics, the input tensor must be one-dimensional.  Since ONNX tensors'
  // first dimension is always the batch size, we ignore the first dimension in
  // our check (e.g. we check for 2, not 1).
  std::vector<size_t> inputDims;
  ExtractTensorDims(graph, softmax.input(0), inputDims);
  if (inputDims.size() != 2)
    return false;

  return true;
}

/**
 * Create a Softplus layer.
 */
inline void SoftmaxSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the SoftPlus layer---there is nothing else to do.
  network.Add<mlpack::Softmax>();
}

} // namespace onnx_mlpack

#endif
