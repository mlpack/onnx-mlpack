/**
 * @file mish_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mish layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MISH_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MISH_IMPL_HPP

#include "mish.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool MishSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // There are no parameters to the ReLU layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& mish = graph.node(nodes[0]);
  if (mish.op_type() != "Mish")
    return false;

  return true;
}

/**
 * Create a Mish layer.
 */
inline void MishSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the Mish layer---there is nothing else to do.
  network.Add<mlpack::Mish>();
}

} // namespace onnx_mlpack

#endif
