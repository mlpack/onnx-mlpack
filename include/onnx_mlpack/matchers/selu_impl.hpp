/**
 * @file selu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the SELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SELU_IMPL_HPP

#include "selu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to an SELU layer.
 */
inline bool SELUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // We only need to check the name of the layer; any alpha value will work.
  const onnx::NodeProto& selu = graph.node(nodes[0]);
  if (selu.op_type() != "Selu")
    return false;

  // Only the default values for the SELU are accepted.
  float alpha = 1.67326319217681884765625f;
  if (!ExtractAttribute(selu, "alpha", alpha))
    return false;
  float gamma = 1.05070102214813232421875f;
  if (!ExtractAttribute(selu, "gamma", gamma))
    return false;

  if (std::abs(alpha - 1.67326319217681884765625f) > 1e-8f)
    return false;
  if (std::abs(gamma - 1.05070102214813232421875f) > 1e-8f)
    return false;

  return true;
}

/**
 * Create a SELU layer with the same metadata as the given ONNX graph.
 */
inline void SELUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  network.Add<mlpack::SELU>();
}

} // namespace onnx_mlpack

#endif
