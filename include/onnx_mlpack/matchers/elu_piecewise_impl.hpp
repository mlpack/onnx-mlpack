/**
 * @file elu_piecewise_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the ELU layer when it is represented
 * as a series of piecewise operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_ELU_PIECEWISE_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_ELU_PIECEWISE_IMPL_HPP

#include "elu_piecewise.hpp"
#include "../extract_scalar.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool ELUPiecewiseSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 9)
    return false;
  for (size_t i = 0; i < nodes.n_elem; ++i)
    if (nodes[i] >= graph.node_size())
      return false;

  // Collect names for our nodes.
  const onnx::NodeProto& gt = graph.node(nodes[0]);
  const onnx::NodeProto& elu = graph.node(nodes[1]);
  const onnx::NodeProto& cond = graph.node(nodes[2]);
  const onnx::NodeProto& negCast = graph.node(nodes[3]);
  const onnx::NodeProto& posCast = graph.node(nodes[4]);
  const onnx::NodeProto& eluMul = graph.node(nodes[5]);
  const onnx::NodeProto& posTerm = graph.node(nodes[6]);
  const onnx::NodeProto& negTerm = graph.node(nodes[7]);
  const onnx::NodeProto& finalOp = graph.node(nodes[8]);

  // Make sure the nodes are correct.
  if (gt.op_type() != "Greater" ||
      elu.op_type() != "Elu" ||
      cond.op_type() != "Not" ||
      negCast.op_type() != "Cast" ||
      posCast.op_type() != "Cast" ||
      eluMul.op_type() != "Mul" ||
      posTerm.op_type() != "Mul" ||
      negTerm.op_type() != "Mul" ||
      finalOp.op_type() != "Add")
    return false;

  // The Greater node must be comparing against a scalar, and its value must be
  // 0.
  double gtVal = DBL_MAX;
  if (!ExtractScalar(graph, gt.input(1), gtVal))
    return false;
  if (gtVal != 0.0)
    return false;

  // The ELU multiplication must be against a scalar (this is the alpha value,
  // which must be greater than 0).
  double alpha = DBL_MAX;
  if (!ExtractEitherScalar(graph, eluMul.input(0), eluMul.input(1), alpha))
    return false;
  if (alpha < 0.0)
    return false;

  // There are no other checks to validate the subgraph.
  return true;
}

/**
 * Create an ELU layer.
 */
inline void ELUPiecewiseSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to extract the alpha value and then we can add the layer.
  double alpha = DBL_MAX;
  const onnx::NodeProto& eluMul = graph.node(nodes[5]);
  if (!ExtractEitherScalar(graph, eluMul.input(0), eluMul.input(1), alpha))
  {
    throw std::runtime_error("ELUPiecewiseSubgraph(): could not extract "
        "alpha value from ONNX Mul node!");
  }

  // We only need to add the ELU layer---there is nothing else to do.
  network.Add<mlpack::ELU>(alpha);
}

} // namespace onnx_mlpack

#endif
