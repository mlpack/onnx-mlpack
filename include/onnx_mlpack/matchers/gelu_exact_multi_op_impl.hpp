/**
 * @file gelu_exact_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the GELU layer when it is represented
 * as a series of operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_EXACT_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_GELU_EXACT_MULTI_OP_IMPL_HPP

#include "gelu_exact_multi_op.hpp"
#include "../extract_scalar.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool GELUExactMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 5)
    return false;
  for (size_t i = 0; i < nodes.n_elem; ++i)
    if (nodes[i] >= graph.node_size())
      return false;

  // Collect names for our nodes.
  const onnx::NodeProto& halfMul = graph.node(nodes[0]);
  const onnx::NodeProto& neg = graph.node(nodes[1]);
  const onnx::NodeProto& divSqrt = graph.node(nodes[2]);
  const onnx::NodeProto& erf = graph.node(nodes[3]);
  const onnx::NodeProto& finalOp = graph.node(nodes[4]);

  // Make sure the nodes are correct.
  if (halfMul.op_type() != "Mul" ||
      neg.op_type() != "Neg" ||
      divSqrt.op_type() != "Mul" ||
      erf.op_type() != "Erfc" ||
      finalOp.op_type() != "Mul")
    return false;

  // The first multiplication needs to be by 0.5.
  double halfMulVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, halfMul.input(0), halfMul.input(1),
      halfMulVal))
    return false;
  if (halfMulVal != 0.5)
    return false;

  // The second multiplication needs to be by (1 / sqrt(2)).
  double divSqrtVal = DBL_MAX;
  if (!ExtractEitherScalar(graph, divSqrt.input(0), divSqrt.input(1),
      divSqrtVal))
    return false;
  if (std::abs(divSqrtVal - (1.0 / std::sqrt(2.0))) > 1e-6)
    return false;

  // There are no other checks to validate the subgraph.
  return true;
}

/**
 * Create a GELU layer.
 */
inline void GELUExactMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // There are no parameters for GELU, so we can just add it.
  network.Add<mlpack::GELUExact>();
}

} // namespace onnx_mlpack

#endif
