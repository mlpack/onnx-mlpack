/**
 * @file elu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the ELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_ELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_ELU_IMPL_HPP

#include "elu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to an ELU layer.
 */
inline bool ELUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  // We only need to check the name of the layer; any alpha value will work.
  const onnx::NodeProto& elu = graph.node(nodes[0]);
  if (elu.op_type() != "Elu")
    return false;

  return true;
}

/**
 * Create a ELU layer with the same metadata as the given ONNX graph.
 */
inline void ELUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& elu = graph.node(nodes[0]);

  // First we have to extract the value of alpha.
  double alpha = 1.0;
  for (size_t i = 0; i < elu.attribute_size(); ++i)
  {
    if (elu.attribute(i).has_name() && elu.attribute(i).name() == "alpha" &&
        elu.attribute(i).has_f())
    {
      alpha = (double) elu.attribute(i).f();
      break;
    }
  }

  // We only need to add the CeLU layer with the right alpha value.
  network.Add<mlpack::ELU>(alpha);
}

} // namespace onnx_mlpack

#endif
