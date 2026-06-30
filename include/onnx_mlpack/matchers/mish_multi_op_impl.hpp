/**
 * @file mish_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mish layer as a series of ONNX
 * operations.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_IMPL_HPP

#include "mish_multi_op.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool MishMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 3)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;
  if (nodes[1] >= graph.node_size())
    return false;
  if (nodes[2] >= graph.node_size())
    return false;

  const onnx::NodeProto& softplus = graph.node(nodes[0]);
  if (softplus.op_type() != "Softplus")
    return false;
  const onnx::NodeProto& tanh = graph.node(nodes[1]);
  if (tanh.op_type() != "Tanh")
    return false;
  const onnx::NodeProto& mul = graph.node(nodes[2]);
  if (mul.op_type() != "Mul")
    return false;

  // We must ensure that the input to the Softplus is also one of the inputs to
  // the Mul.
  if (softplus.input(0) != mul.input(0) && softplus.input(0) != mul.input(1))
    return false;

  // We cannot have broadcasting in the mul operation, so we need to ensure that
  // both inputs have the same dimensions.
  std::vector<size_t> mulADims;
  std::vector<size_t> mulBDims;
  ExtractTensorDims(graph, mul.input(0), mulADims);
  ExtractTensorDims(graph, mul.input(1), mulBDims);

  // Make sure we found the initializers.
  if (mulADims.size() == 0 || mulBDims.size() == 0)
    return false;

  // Make sure they have the same number of dimensions: if so, we are not
  // broadcasting.
  if (mulADims.size() != mulBDims.size())
    return false;

  return true;
}

/**
 * Create a Mish layer.
 */
inline void MishMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the Mish layer---there is nothing else to do.
  network.Add<mlpack::Mish>();
}

} // namespace onnx_mlpack

#endif
