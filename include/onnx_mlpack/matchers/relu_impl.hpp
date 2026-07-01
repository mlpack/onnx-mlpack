/**
 * @file relu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the ReLU layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_RELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_RELU_IMPL_HPP

#include "relu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool ReLUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // There are no parameters to the ReLU layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& relu = graph.node(nodes[0]);
  if (relu.op_type() != "Relu")
    return false;

  return true;
}

/**
 * Create a ReLU layer.
 */
inline void ReLUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the ReLU layer---there is nothing else to do.
  network.Add<mlpack::ReLU>();
}

} // namespace onnx_mlpack

#endif
