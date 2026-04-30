/**
 * @file leaky_relu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LeakyReLU layer.
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
  double alpha = 0.01;
  for (size_t i = 0; i < relu.attribute_size(); ++i)
  {
    if (relu.attribute(i).has_name() &&
        relu.attribute(i).name() == "alpha" &&
        relu.attribute(i).has_f())
    {
      alpha = (double) relu.attribute(i).f();
    }
  }

  network.Add<mlpack::LeakyReLU>(alpha);
}

} // namespace onnx_mlpack

#endif
