/**
 * @file hard_swish_sigmoid_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSwish layer using an ONNX
 * HardSigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SWISH_SIGMOID_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SWISH_SIGMOID_IMPL_HPP

#include "hard_swish_sigmoid.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a HardSwish layer.
 */
inline bool HardSwishSigmoidSubgraph::Validate(
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

  // This must match the HardSwish parameters nearly exactly.
  if (std::abs(alpha - 0.166667) > 1e-6)
    return false;
  if (std::abs(beta - 0.5) > 1e-6)
    return false;

  return true;
}

/**
 * Create a HardSwish layer.
 */
inline void HardSwishSigmoidSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the HardSwish layer.
  network.Add<mlpack::HardSwish>();
}

} // namespace onnx_mlpack

#endif
