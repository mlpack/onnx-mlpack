/**
 * @file gelu_exact_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the exact GELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_EXACT_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_GELU_EXACT_IMPL_HPP

#include "gelu_exact.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to an exact GELU layer.
 */
inline bool GELUExactSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
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

  // This is the exact version only.
  if (approx != "none")
    return false;

  return true;
}

/**
 * Create an exact GELU layer.
 */
inline void GELUExactSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the GELU layer.
  network.Add<mlpack::GELUExact>();
}

} // namespace onnx_mlpack

#endif
