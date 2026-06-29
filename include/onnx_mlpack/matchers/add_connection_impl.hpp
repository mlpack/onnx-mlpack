/**
 * @file add_connection_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match an Add ONNX operation that represents
 * when the outputs of two other layers need to be added together.
 */
#ifndef ONNX_MLPACK_MATCHERS_ADD_CONNECTION_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_ADD_CONNECTION_IMPL_HPP

#include "add_connection.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to an ADDITION connection in
 * a DAGNetwork.
 */
inline bool AddConnectionSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  const onnx::NodeProto& add = graph.node(nodes[0]);
  if (add.op_type() != "Add")
    return false;

  // We need to ensure that the two inputs of the Add node are both outputs of
  // other operations, and we must also ensure that the input tensors have the
  // same shape.  Lastly, we have to ensure that the output tensor is used as
  // input to another node.

  if (add.input_size() != 2)
    return false;

  const std::string& aInput = add.input(0);
  const std::string& bInput = add.input(1);

  // The input tensors must be distinct.
  if (aInput == bInput)
    return false;

  // These tensors can't have initializers.
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == aInput)
      return false;

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bInput)
      return false;
  }

  // They must be the output of some other node.
  bool aIsOutput = false;
  bool bIsOutput = false;
  for (size_t i = 0; i < graph.node_size(); ++i)
  {
    for (size_t j = 0; j < graph.node(i).output_size(); ++j)
    {
      if (graph.node(i).output(j) == aInput)
        aIsOutput = true;
      else if (graph.node(i).output(j) == bInput)
        bIsOutput = true;
    }
  }

  if (!aIsOutput)
    return false;
  if (!bIsOutput)
    return false;

  std::vector<size_t> aDims, bDims;

  // Get the inferred dimensions of the tensors and make sure they are the same.
  for (size_t i = 0; i < graph.value_info_size(); ++i)
  {
    const onnx::ValueInfoProto& v = graph.value_info(i);
    if (v.has_name() && v.name() == aInput && v.has_type() &&
        v.type().has_tensor_type() && v.type().tensor_type().has_shape())
    {
      for (size_t i = 0; i < v.type().tensor_type().shape().dim_size(); ++i)
      {
        if (v.type().tensor_type().shape().dim(i).has_dim_value())
          aDims.push_back(v.type().tensor_type().shape().dim(i).dim_value());
        else
          aDims.push_back(0 /* dummy placeholder */);
      }
    }

    if (v.has_name() && v.name() == bInput && v.has_type() &&
        v.type().has_tensor_type() && v.type().tensor_type().has_shape())
    {
      for (size_t i = 0; i < v.type().tensor_type().shape().dim_size(); ++i)
      {
        if (v.type().tensor_type().shape().dim(i).has_dim_value())
          bDims.push_back(v.type().tensor_type().shape().dim(i).dim_value());
        else
          bDims.push_back(0 /* dummy placeholder */);
      }
    }
  }

  if (aDims.size() != bDims.size())
    return false;
  for (size_t d = 0; d < aDims.size(); ++d)
    if (aDims[d] != bDims[d])
      return false;

  // Lastly, make sure the output is used somewhere.
  const std::string& outputName = add.output(0);
  bool outputFound = false;
  for (size_t i = 0; i < graph.node_size(); ++i)
  {
    for (size_t j = 0; j < graph.node(i).input_size(); ++j)
    {
      if (graph.node(i).input(j) == outputName)
      {
        outputFound = true;
        break;
      }
    }
  }

  if (!outputFound)
    return false;

  return true;
}

/**
 * Dummy method: we don't need to add a layer to the network.
 */
inline void AddConnectionSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We actually don't want to add a layer here; instead, the main subgraph
  // conversion network will make connections between the two input layers of
  // this layer and the output of this layer.
  return;
}

} // namespace onnx_mlpack

#endif
