/**
 * @file sigmoid_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Sigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SIGMOID_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SIGMOID_IMPL_HPP

#include "sigmoid.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Sigmoid layer.
 */
inline bool SigmoidSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // There are no parameters to the sigmoid layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& sigmoid = graph.node(nodes[0]);
  if (sigmoid.op_type() != "Sigmoid")
    return false;

  return true;
}

/**
 * Create a Sigmoid layer.
 */
inline void SigmoidSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the Sigmoid layer---there is nothing else to do.
  network.Add<mlpack::Sigmoid>();
}

} // namespace onnx_mlpack

#endif
