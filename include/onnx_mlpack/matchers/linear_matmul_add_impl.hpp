/**
 * @file linear_matmul_add_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Linear layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_IMPL_HPP

#include "linear_matmul_add.hpp"

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
  if (nodes[0] >= graph.node_size())
    return false;
  if (nodes[1] >= graph.node_size())
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
  std::vector<size_t> weightDims, add1Dims, add2Dims;
  ExtractTensorDims(graph, matmul.input(1), weightDims, true);
  ExtractTensorDims(graph, add.input(0), add1Dims, true);
  ExtractTensorDims(graph, add.input(1), add2Dims, true);
  if (weightDims.size() != 2)
    return false;
  if (add1Dims.size() == 0 && add2Dims.size() == 0)
    return false;
  if (add1Dims.size() == 1 || add2Dims.size() == 1)
    return true;

  return false;
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
    std::vector<mlpack::Layer<>*>& layers) const
{
  // We have already concluded that the weights of the operation must be the
  // second input of the MatMul node, and the biases are the initialized input
  // of the Add node.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  const onnx::NodeProto& add = graph.node(nodes[1]);

  mlpack::Linear<>* l = dynamic_cast<mlpack::Linear<>*>(layers[0]);
  l->Weight() = TensorToArma(graph, matmul.input(1));

  std::vector<size_t> biasADims, biasBDims;
  ExtractTensorDims(graph, add.input(0), biasADims, true);
  ExtractTensorDims(graph, add.input(1), biasBDims, true);
  if (biasADims.size() == 1)
  {
    l->Bias() = TensorToArma(graph, add.input(0)).t();
  }
  else if (biasBDims.size() == 1)
  {
    l->Bias() = TensorToArma(graph, add.input(1)).t();
  }
  else
  {
    throw std::runtime_error("LinearMatMulAddSubgraph::TransferWeights(): "
        "could not extract bias vector!");
  }
}

} // namespace onnx_mlpack

#endif
