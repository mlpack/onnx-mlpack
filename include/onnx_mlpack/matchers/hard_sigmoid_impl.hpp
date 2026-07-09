/**
 * @file hard_sigmoid_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSigmoid layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SIGMOID_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SIGMOID_IMPL_HPP

#include "hard_sigmoid.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a HardSigmoid layer.
 * Only default values for alpha and beta are supported.
 */
inline bool HardSigmoidSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  const onnx::NodeProto& hs = graph.node(nodes[0]);
  if (hs.op_type() != "HardSigmoid")
    return false;

  // The mlpack HardSigmoid layer is hard-coded to have alpha = 0.2 and beta =
  // 0.5, so we must ensure we have those values only.
  float alpha = 0.2f;
  float beta = 0.5f;
  if (!ExtractAttribute(hs, "alpha", alpha))
    return false;
  if (!ExtractAttribute(hs, "beta", beta))
    return false;

  // We must have default parameters.
  if (alpha != 0.2f)
    return false;
  if (beta != 0.5f)
    return false;

  return true;
}

/**
 * Create a HardSigmoid layer.
 */
inline void HardSigmoidSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the HardSigmoid layer.
  network.Add<mlpack::HardSigmoid>();
}

} // namespace onnx_mlpack

#endif
