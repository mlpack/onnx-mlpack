/**
 * @file celu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the CELU layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_CELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_CELU_IMPL_HPP

#include "celu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a CELU layer.
 */
inline bool CELUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // We only need to check that we have a Celu node; any alpha value will be
  // fine.
  const onnx::NodeProto& celu = graph.node(nodes[0]);
  if (celu.op_type() != "Celu")
    return false;

  return true;
}

/**
 * Create a CELU layer with the same metadata as the given ONNX graph.
 */
inline void CELUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& celu = graph.node(nodes[0]);

  // First we have to extract the value of alpha.
  float alpha = 1.0f;
  if (!ExtractAttribute(celu, "alpha", alpha))
  {
    throw std::runtime_error("CELUSubgraph::Convert(): could not extract "
        "'alpha' attribute!");
  }

  // We only need to add the CELU layer with the right alpha value.
  network.Add<mlpack::CELU>((double) alpha);
}

} // namespace onnx_mlpack

#endif
