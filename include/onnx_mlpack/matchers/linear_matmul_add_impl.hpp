/**
 * @file linear_matmul_add_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Linear layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_IMPL_HPP

#include "linear_matmul_add.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Linear layer.
 */
inline bool LinearMatMulAddSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 2)
    return false;
  if (nodes[0] > graph.node_size())
    return false;
  if (nodes[1] > graph.node_size())
    return false;

  // Sanity check the attributes of the gemm to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  const onnx::NodeProto& add = graph.node(nodes[1]);
  if (matmul.op_type() != "MatMul")
    return false;
  if (add.op_type() != "Add")
    return false;

  // We require that the second input parameter of the MatMul is fully
  // initialized, and that one of the two input parameters of the Add is fully
  // initialized.
  const std::string bName = matmul.input(1);
  const std::string addName1 = add.input(0);
  const std::string addName2 = add.input(1);
  bool matmulInitializer = false;
  size_t addInitializersFound = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      matmulInitializer = true;
    }

    if (graph.initializer(i).has_name() &&
        (graph.initializer(i).name() == addName1 ||
         graph.initializer(i).name() == addName2) &&
        graph.initializer(i).dims_size() == 1)
    {
      addInitializersFound += 1;
    }
  }

  if (!matmulInitializer || (addInitializersFound != 1))
    return false;

  return true;
}

/**
 * Create a Linear layer with the same metadata as the given ONNX graph.
 */
inline void LinearMatMulAddSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We can extract the output size of the matrix from the weights that we are
  // multiplying.

  size_t outputDims = 0;
  const onnx::NodeProto& matmul = graph.node(nodes[0]);

  // The second dimension of B is the output size.
  const std::string bName = matmul.input(1);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      outputDims = graph.initializer(i).dims(1);
    }
  }

  if (outputDims == 0)
  {
    throw std::runtime_error("LinearMatMulAddSubgraph::Convert(): cannot "
        "infer output size of ONNX Gemm operation!");
  }

  network.Add<mlpack::Linear>(outputDims);
}

inline void LinearMatMulAddSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the
  // second input of the MatMul node, and the biases are the initialized input
  // of the Add node.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  const onnx::NodeProto& add = graph.node(nodes[1]);

  const std::string wName = matmul.input(1);
  const std::string addName1 = add.input(0);
  const std::string addName2 = add.input(1);

  bool weightsDone = false;
  bool biasesDone = false;
  mlpack::Linear<>* l = dynamic_cast<mlpack::Linear<>*>(layer);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == wName &&
        graph.initializer(i).dims_size() == 2)
    {
      l->Weight() = TensorToArma(graph.initializer(i));
      weightsDone = true;
    }

    if (graph.initializer(i).has_name() &&
        (graph.initializer(i).name() == addName1 ||
         graph.initializer(i).name() == addName2) &&
        graph.initializer(i).dims_size() == 1)
    {
      l->Bias() = TensorToArma(graph.initializer(i)).t();
      biasesDone = true;
    }
  }

  if (!weightsDone || !biasesDone)
  {
    throw std::runtime_error("LinearMatMulAddSubgraph::TransferWeights(): "
        "failed to find weight tensor in ONNX graph!");
  }
}

} // namespace onnx_mlpack

#endif
