/**
 * @file celu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the CELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_CELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_CELU_IMPL_HPP

#include "celu.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a CELU layer.
 */
inline bool CELUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  // We only need to check that we have a Celu node; any alpha value will be
  // fine.
  const onnx::NodeProto& celu = graph.node(nodes[0]);
  if (celu.op_type() != "Celu")
    return false;

  return true;
}

/**
 * Create a CELU layer with the same metadata as the given ONNX graph.
 */
inline void CELUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& celu = graph.node(nodes[0]);

  // First we have to extract the value of alpha.
  double alpha = 1.0;
  for (size_t i = 0; i < celu.attribute_size(); ++i)
  {
    if (celu.attribute(i).has_name() && celu.attribute(i).name() == "alpha" &&
        celu.attribute(i).has_f())
    {
      alpha = (double) celu.attribute(i).f();
      break;
    }
  }

  // We only need to add the CELU layer with the right alpha value.
  network.Add<mlpack::CELU>(alpha);
}

} // namespace onnx_mlpack

#endif
