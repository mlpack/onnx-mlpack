/**
 * @file tanh_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Tanh layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_TANH_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_TANH_IMPL_HPP

#include "tanh.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Tanh layer.
 */
inline bool TanhSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  const onnx::NodeProto& tanh = graph.node(nodes[0]);
  if (tanh.op_type() != "Tanh")
    return false;

  return true;
}

/**
 * Create a Tanh layer.
 */
inline void TanhSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the TanH layer---there is nothing else to do.
  network.Add<mlpack::TanH>();
}

} // namespace onnx_mlpack

#endif
