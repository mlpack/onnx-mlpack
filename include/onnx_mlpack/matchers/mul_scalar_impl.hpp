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
#include "../tensor_to_arma.hpp"

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
  const std::string& aName = mul.input(0);
  const std::string& bName = mul.input(1);
  size_t aElems = 0;
  size_t bElems = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == aName &&
        graph.initializer(i).dims_size() >= 1)
    {
      aElems = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        aElems *= graph.initializer(i).dims(j);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() >= 1)
    {
      bElems = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        bElems *= graph.initializer(i).dims(j);
    }
  }

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
  const std::string& aName = mul.input(0);
  const std::string& bName = mul.input(1);
  size_t aElems = 0;
  size_t aIndex = 0;
  size_t bElems = 0;
  size_t bIndex = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == aName &&
        graph.initializer(i).dims_size() >= 1)
    {
      aElems = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        aElems *= graph.initializer(i).dims(j);
      aIndex = i;
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() >= 1)
    {
      bElems = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        bElems *= graph.initializer(i).dims(j);
      bIndex = i;
    }
  }

  arma::mat scalarMat;
  scalarMat = TensorToArma(graph.initializer(aElems == 1 ? aIndex : bIndex),
      true);
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
