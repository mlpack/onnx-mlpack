/**
 * @file linear_no_bias_gemm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_IMPL_HPP

#include "linear_no_bias_matmul.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool LinearNoBiasMatMulSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MatMul to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  if (matmul.op_type() != "MatMul")
    return false;

  // We require that the second input parameter, the weights, are fully
  // initialized.
  std::vector<size_t> weightDims;
  ExtractTensorDims(graph, matmul.input(1), weightDims, true);
  if (weightDims.size() != 2)
    return false;

  return true;
}

/**
 * Create a LinearNoBias layer with the same metadata as the given ONNX graph.
 */
inline void LinearNoBiasMatMulSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a MatMul, we don't have anything to do
  // other than create the LinearNoBias layer.  However, we must first compute
  // the number of output nodes using the shape of the graph.
  //
  // There are a few possibilities: if C is specified, we can steal the size
  // from there.  If C is not specified, then we must infer the size based on
  // the shapes of A and B (and the settings of transA and transB).

  const onnx::NodeProto& matmul = graph.node(nodes[0]);

  // The second dimension of B is the output size.
  std::vector<size_t> weightDims;
  ExtractTensorDims(graph, matmul.input(1), weightDims, true);
  if (weightDims[1] == 0)
  {
    throw std::runtime_error("LinearNoBiasMatMulSubgraph::Convert(): cannot "
        "infer output size of ONNX MatMul operation!");
  }

  network.Add<mlpack::LinearNoBias>(weightDims[1]);
}

inline void LinearNoBiasMatMulSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the MatMul operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  mlpack::LinearNoBias<>* l = dynamic_cast<mlpack::LinearNoBias<>*>(layers[0]);
  l->Parameters() = TensorToArma(graph, matmul.input(1));
}

} // namespace onnx_mlpack

#endif
