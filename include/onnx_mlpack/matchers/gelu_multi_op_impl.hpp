/**
 * @file gelu_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the approximate GELU layer when it is
 * represented as a series of operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_GELU_MULTI_OP_IMPL_HPP

#include "gelu_multi_op.hpp"
#include "../extract_scalar.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool GELUMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 8)
    return false;
  for (size_t i = 0; i < nodes.n_elem; ++i)
    if (nodes[i] >= graph.node_size())
      return false;

  // Collect names for our nodes.
  const onnx::NodeProto& halfMul = graph.node(nodes[0]);
  const onnx::NodeProto& pow = graph.node(nodes[1]);
  const onnx::NodeProto& powMul = graph.node(nodes[2]);
  const onnx::NodeProto& add = graph.node(nodes[3]);
  const onnx::NodeProto& tanhInner = graph.node(nodes[4]);
  const onnx::NodeProto& tanh = graph.node(nodes[5]);
  const onnx::NodeProto& tanhAdd = graph.node(nodes[6]);
  const onnx::NodeProto& outerMul = graph.node(nodes[7]);

  // Make sure the nodes are correct.
  if (halfMul.op_type() != "Mul" ||
      pow.op_type() != "Pow" ||
      powMul.op_type() != "Mul" ||
      add.op_type() != "Add" ||
      tanhInner.op_type() != "Mul" ||
      tanh.op_type() != "Tanh" ||
      tanhAdd.op_type() != "Add" ||
      outerMul.op_type() != "Mul")
    return false;

  // The first multiplication needs to be by 0.5.
  double halfMulVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, halfMul.input(0), halfMul.input(1),
      halfMulVal))
    return false;
  if (halfMulVal != 0.5)
    return false;

  // The power must be 3.
  double powVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, pow.input(0), pow.input(1), powVal))
    return false;
  if (powVal != 3.0)
    return false;

  // The second multiplication needs to be by 0.044715.
  double powMulVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, powMul.input(0), powMul.input(1), powMulVal))
    return false;
  if (std::abs(powMulVal - 0.044715) > 1e-6)
    return false;

  // The third multiplication must be by sqrt(2 / pi).
  double tanhInnerVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, tanhInner.input(0), tanhInner.input(1),
      tanhInnerVal))
    return false;
  if (std::abs(tanhInnerVal - std::sqrt(2.0 / M_PI)) > 1e-6)
    return false;

  // The inner addition must be with a scalar 1.
  double addVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, tanhAdd.input(0), tanhAdd.input(1), addVal))
    return false;
  if (addVal != 1.0)
    return false;

  // There are no other checks to validate the subgraph.
  return true;
}

/**
 * Create a GELU layer.
 */
inline void GELUMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // There are no parameters for GELU, so we can just add it.
  network.Add<mlpack::GELU>();
}

} // namespace onnx_mlpack

#endif
