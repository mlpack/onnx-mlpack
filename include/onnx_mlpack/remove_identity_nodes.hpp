/**
 * @file remove_identity_nodes.hpp
 * @author Ryan Curtin
 *
 * If an ONNX graph has Identity operations, they can be trivially removed (they
 * don't do anything).
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_REMOVE_IDENTITY_NODES_HPP
#define ONNX_MLPACK_REMOVE_IDENTITY_NODES_HPP

namespace onnx_mlpack {

inline void RemoveIdentityNodes(onnx::GraphProto& graph)
{
  // Iterate over all the nodes in the graph and look for Identity operations.
  std::set<size_t> nodesToRemove;

  // For each Identity node we will remove, this holds both the input and output
  // tensor names.  (Elsewhere in the network, we will have to replace
  // references to the output tensor with references to the input tensor.)
  std::unordered_map<std::string, std::string> tensorReplacements;
  for (size_t n = 0; n < graph.node_size(); ++n)
  {
    const onnx::NodeProto& node = graph.node(n);
    if (node.op_type() == "Identity")
    {
      nodesToRemove.insert(n);
      tensorReplacements[node.output(0)] = node.input(0);
    }
  }

  // Now actually remove the nodes we don't need.  Protobuf only lets us remove
  // everything, then re-add it.
  for (size_t n = graph.node_size(); n > 0; --n)
    if (nodesToRemove.count(n - 1) > 0)
      graph.mutable_node()->DeleteSubrange(n - 1, 1);

  // Update inputs for other nodes.
  for (size_t n = 0; n < graph.node_size(); ++n)
  {
    onnx::NodeProto& node = *graph.mutable_node(n);
    for (size_t i = 0; i < node.input_size(); ++i)
      if (tensorReplacements.count(node.input(i)) > 0)
        node.set_input(i, tensorReplacements[node.input(i)]);
  }

  // Update graph outputs, if needed.
  for (size_t i = 0; i < graph.output_size(); ++i)
  {
    if (graph.output(i).has_name() &&
        tensorReplacements.count(graph.output(i).name()) > 0)
    {
      graph.mutable_output(i)->set_name(
          tensorReplacements[graph.output(i).name()]);
    }
  }
}

} // namespace onnx_mlpack

#endif
