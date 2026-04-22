/**
 * @file swish_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Swish layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SWISH_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SWISH_IMPL_HPP

#include "swish.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Swish layer.
 */
inline bool SwishSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  const onnx::NodeProto& swish = graph.node(nodes[0]);
  if (swish.op_type() != "Softplus")
    return false;

  // mlpack does not allow the alpha parameter to be set, so if it's anything
  // different than 1, then we can't match this node.
  double alpha = 1.0;
  for (size_t i = 0; i < swish.attribute_size(); ++i)
  {
    if (swish.attribute(i).has_name() && swish.attribute(i).name() == "alpha" &&
        swish.attribute(i).has_f())
    {
      alpha = (double) swish.attribute(i).f();
      break;
    }
  }

  if (alpha != 1.0)
    return false;

  return true;
}

/**
 * Create a Swish layer.
 */
inline void SwishSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the Swish layer---there is nothing else to do.
  network.Add<mlpack::Swish>();
}

} // namespace onnx_mlpack

#endif
