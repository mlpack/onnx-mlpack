/**
 * @file hard_sigmoid_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the HardSigmoid layer when it is
 * represented as a series of operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SIGMOID_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SIGMOID_MULTI_OP_IMPL_HPP

#include "hard_sigmoid_multi_op.hpp"
#include "../extract_scalar.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool HardSigmoidMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 3)
    return false;
  for (size_t i = 0; i < nodes.n_elem; ++i)
    if (nodes[i] >= graph.node_size())
      return false;

  // Collect names for our nodes.
  const onnx::NodeProto& add = graph.node(nodes[0]);
  const onnx::NodeProto& clip = graph.node(nodes[1]);
  const onnx::NodeProto& mul = graph.node(nodes[2]);

  // Make sure the nodes are correct.
  if (add.op_type() != "Add" ||
      clip.op_type() != "Clip" ||
      mul.op_type() != "Mul")
    return false;

  // Collect the constant values.
  double addVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, add.input(0), add.input(1), addVal))
    return false;

  double clipLower = DBL_MAX;
  double clipUpper = DBL_MAX;
  if (!ExtractScalar(graph, clip.input(1), clipLower))
    return false;
  if (!ExtractScalar(graph, clip.input(2), clipUpper))
    return false;

  double mulVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, mul.input(0), mul.input(1), mulVal))
    return false;

  // Ensure that the values are consistent.
  if (clipLower != 0.0)
    return false;
  if (std::abs(2 * addVal - clipUpper) > 1e-6)
    return false;
  if (std::abs(1 / clipUpper - mulVal) > 1e-6)
    return false;

  // mlpack only supports addVal = 2.5.
  if (addVal != 2.5)
    return false;

  // There are no other checks to validate the subgraph.
  return true;
}

/**
 * Create a HardSigmoid layer.
 */
inline void HardSigmoidMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // There are no parameters for GELU, so we can just add it.
  network.Add<mlpack::HardSigmoid>();
}

} // namespace onnx_mlpack

#endif
