/**
 * @file gelu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the approximate GELU layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_GELU_IMPL_HPP

#include "gelu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to an approximate GELU
 * layer.
 */
inline bool GELUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  const onnx::NodeProto& gelu = graph.node(nodes[0]);
  if (gelu.op_type() != "Gelu")
    return false;

  // Ensure that this is the exact form.
  std::string approx = "none";
  if (!ExtractAttribute(gelu, "approximate", approx))
    return false;

  // This is the approximate version only.
  if (approx != "tanh")
    return false;

  return true;
}

/**
 * Create an exact GELU layer.
 */
inline void GELUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the GELU layer.
  network.Add<mlpack::GELU>();
}

} // namespace onnx_mlpack

#endif
