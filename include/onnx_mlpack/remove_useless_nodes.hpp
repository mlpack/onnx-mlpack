/**
 * @file remove_useless_nodes.hpp
 * @author Ryan Curtin
 *
 * If an ONNX graph has operations that don't actually do anything, they can be
 * trivially removed.
 */
#ifndef ONNX_MLPACK_REMOVE_USELESS_NODES_HPP
#define ONNX_MLPACK_REMOVE_USELESS_NODES_HPP

#include "tensor_to_arma.hpp"

namespace onnx_mlpack {

inline void RemoveUselessNodes(onnx::GraphProto& graph)
{
  // Iterate over all the nodes in the graph and look for Identity operations.
  std::set<size_t> nodesToRemove;
  std::set<std::string> tensorsToRemove;
  std::unordered_map<std::string, std::string> tensorsToReplace;

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
    else if (node.op_type() == "Mul" || node.op_type() == "Add")
    {
      std::cout << "check if we can remove node " << n << " (" << node.op_type() << ")\n";

      // We can remove a Mul node if we are multiplying by all 1 values, and an
      // Add node if we are adding only zero values.
      const double targetValue = (node.op_type() == "Mul") ? 1.0 : 0.0;

      const std::string& aName = node.input(0);
      const std::string& bName = node.input(1);
      bool canRemoveA = false;
      bool canRemoveB = false;
      for (size_t i = 0; i < graph.initializer_size(); ++i)
      {
        if (graph.initializer(i).has_name() &&
            graph.initializer(i).name() == aName &&
            graph.initializer(i).dims_size() >= 1)
        {
          arma::mat vals = TensorToArma(graph.initializer(i), true);
          if (all(all(vals == targetValue)))
            canRemoveA = true;
        }

        if (graph.initializer(i).has_name() &&
            graph.initializer(i).name() == bName &&
            graph.initializer(i).dims_size() >= 1)
        {
          arma::mat vals = TensorToArma(graph.initializer(i), true);
          if (all(all(vals == targetValue)))
            canRemoveB = true;
        }
      }

      // If we can remove only one of the tensors, then we can remove the whole
      // thing.
      if (canRemoveA && !canRemoveB)
      {
        nodesToRemove.insert(n);
        tensorReplacements[node.output(0)] = bName;
      }
      else if (canRemoveB && !canRemoveA)
      {
        nodesToRemove.insert(n);
        tensorReplacements[node.output(0)] = aName;
      }
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
