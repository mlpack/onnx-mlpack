/**
 * @file mul_scalar_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mul ONNX node when it is
 * multiplying the input by a scalar..
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_MUL_SCALAR_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MUL_SCALAR_IMPL_HPP

#include "mul_scalar.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a CELU layer.
 */
inline bool MulScalarSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // We only need to check that we have a Mul node; any alpha value will be
  // fine.
  const onnx::NodeProto& mul = graph.node(nodes[0]);
  if (mul.op_type() != "Mul")
    return false;

  // One of the multiplication operands must be a scalar whose value we know.
  std::vector<size_t> aDims, bDims;
  ExtractTensorDims(graph, mul.input(0), aDims, true);
  ExtractTensorDims(graph, mul.input(1), bDims, true);
  size_t aElems = (aDims.size() == 0) ? 0 : aDims[0];
  size_t bElems = (bDims.size() == 0) ? 0 : bDims[0];
  for (size_t i = 1; i < aDims.size(); ++i)
    aElems *= aDims[i];
  for (size_t i = 1; i < bDims.size(); ++i)
    bElems *= bDims[i];

  // At least one input must be one-element.
  if (aElems != 1 && bElems != 1)
    return false;

  return true;
}

/**
 * Create a Scale layer with the same metadata as the given ONNX graph.
 */
inline void MulScalarSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& mul = graph.node(nodes[0]);

  // Determine which input has the scalar value.
  std::vector<size_t> aDims, bDims;
  ExtractTensorDims(graph, mul.input(0), aDims, true);
  ExtractTensorDims(graph, mul.input(1), bDims, true);
  size_t aElems = (aDims.size() == 0) ? 0 : aDims[0];
  size_t bElems = (bDims.size() == 0) ? 0 : bDims[0];
  for (size_t i = 1; i < aDims.size(); ++i)
    aElems *= aDims[i];
  for (size_t i = 1; i < bDims.size(); ++i)
    bElems *= bDims[i];

  arma::mat scalarMat = TensorToArma(graph,
      (aElems == 1) ? mul.input(0) : mul.input(1), true);
  if (scalarMat.n_elem != 1)
  {
    throw std::runtime_error("MulScalarSubgraph::Convert(): got multiple values"
        " for scalar tensor; it should have only one element!");
  }

  const double scaleFactor = scalarMat[0];

  // We only need to add the Scale layer with the right alpha value.
  network.Add<mlpack::Scale>(scaleFactor);
}

} // namespace onnx_mlpack

#endif
