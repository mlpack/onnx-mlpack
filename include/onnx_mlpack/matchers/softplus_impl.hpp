/**
 * @file softplus_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Sigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SOFTPLUS_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SOFTPLUS_IMPL_HPP

#include "softplus.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Softplus layer.
 */
inline bool SoftplusSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  // There are no parameters to the Softplus layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& softplus = graph.node(nodes[0]);
  if (softplus.op_type() != "Softplus")
    return false;

  return true;
}

/**
 * Create a Softplus layer.
 */
inline void SoftplusSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the SoftPlus layer---there is nothing else to do.
  network.Add<mlpack::SoftPlus>();
}

} // namespace onnx_mlpack

#endif
