/**
 * @file gelu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the approximate GELU layer.
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
  if (nodes[0] > graph.node_size())
    return false;

  const onnx::NodeProto& gelu = graph.node(nodes[0]);
  if (gelu.op_type() != "Gelu")
    return false;

  // Ensure that this is the exact form.
  std::string approx = "none";
  for (size_t i = 0; i < gelu.attribute_size(); ++i)
  {
    if (gelu.attribute(i).has_name() &&
        gelu.attribute(i).name() == "approximate" &&
        gelu.attribute(i).has_s())
    {
      approx = gelu.attribute(i).s();
    }
  }

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
