/**
 * @file hard_swish_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSwish layer using an ONNX
 * HardSwish layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SWISH_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SWISH_IMPL_HPP

#include "hard_swish.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a HardSwish layer.
 */
inline bool HardSwishSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  const onnx::NodeProto& hs = graph.node(nodes[0]);
  if (hs.op_type() != "HardSwish")
    return false;

  return true;
}

/**
 * Create a HardSwish layer.
 */
inline void HardSwishSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the HardSwish layer.
  network.Add<mlpack::HardSwish>();
}

} // namespace onnx_mlpack

#endif
