/**
 * @file leaky_relu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LeakyReLU layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LEAKY_RELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LEAKY_RELU_IMPL_HPP

#include "leaky_relu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool LeakyReLUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Any alpha value is sufficient for us to match.
  const onnx::NodeProto& relu = graph.node(nodes[0]);
  if (relu.op_type() != "LeakyRelu")
    return false;

  return true;
}

/**
 * Create a LeakyReLU layer.
 */
inline void LeakyReLUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& relu = graph.node(nodes[0]);

  // First we have to extract the correct alpha value.
  float alpha = 0.01f;
  if (!ExtractAttribute(relu, "alpha", alpha))
  {
    throw std::runtime_error("LeakyReLUSubgraph::Convert(): could not extract "
        "'alpha' attribute!");
  }

  network.Add<mlpack::LeakyReLU>((double) alpha);
}

} // namespace onnx_mlpack

#endif
