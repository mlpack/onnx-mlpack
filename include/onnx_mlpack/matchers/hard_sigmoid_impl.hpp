/**
 * @file hard_sigmoid_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SIGMOID_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SIGMOID_IMPL_HPP

#include "hard_sigmoid.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a HardSigmoid layer.
 * Only default values for alpha and beta are supported.
 */
inline bool HardSigmoidSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  const onnx::NodeProto& hs = graph.node(nodes[0]);
  if (hs.op_type() != "HardSigmoid")
    return false;

  // The mlpack HardSigmoid layer is hard-coded to have alpha = 0.2 and beta =
  // 0.5, so we must ensure we have those values only.
  double alpha = 0.2;
  double beta = 0.5;
  for (size_t i = 0; i < hs.attribute_size(); ++i)
  {
    if (hs.attribute(i).has_name() &&
        hs.attribute(i).name() == "alpha" &&
        hs.attribute(i).has_f())
    {
      alpha = (double) hs.attribute(i).f();
    }

    if (hs.attribute(i).has_name() &&
        hs.attribute(i).name() == "beta" &&
        hs.attribute(i).has_f())
    {
      beta = (double) hs.attribute(i).f();
    }
  }

  // We must have default parameters.
  if (alpha != 0.2)
    return false;
  if (beta != 0.5)
    return false;

  return true;
}

/**
 * Create a HardSigmoid layer.
 */
inline void HardSigmoidSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the HardSigmoid layer.
  network.Add<mlpack::HardSigmoid>();
}

} // namespace onnx_mlpack

#endif
