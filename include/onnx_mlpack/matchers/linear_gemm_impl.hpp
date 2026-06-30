/**
 * @file linear_gemm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Linear layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_GEMM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_GEMM_IMPL_HPP

#include "linear_gemm.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Linear layer.
 */
inline bool LinearGemmSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the gemm to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  if (gemm.op_type() != "Gemm")
    return false;
  if (gemm.input_size() != 3) // we must use C for this matcher.
    return false;

  // We cannot accept when beta is 0: then there is no bias.
  float beta = 1.0f; // default according to ONNX spec
  if (!ExtractAttribute(gemm, "beta", beta))
    return false;
  if (beta == 0.0f)
    return false;

  // We require that the second and third input parameters (weights and biases)
  // are fully initialized.
  std::vector<size_t> bDims, cDims;
  ExtractTensorDims(graph, gemm.input(1), bDims);
  ExtractTensorDims(graph, gemm.input(2), cDims);
  if (bDims.size() == 0 || cDims.size() == 0)
    return false;

  // The alpha attribute must be set to 1; the Linear layer doesn't support
  // constant scaling.
  float alpha = 1.0f;
  if (!ExtractAttribute(gemm, "alpha", alpha))
    return false;
  if (alpha != 1.0f)
    return false;

  return true;
}

/**
 * Create a Linear layer with the same metadata as the given ONNX graph.
 */
inline void LinearGemmSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a Gemm, we don't have anything to do
  // other than create the Linear layer.  However, we must first compute the
  // number of output nodes using the shape of the graph.  We can do this by
  // taking the size of C (the biases).

  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  std::vector<size_t> biasDims;
  ExtractTensorDims(graph, gemm.input(2), biasDims);
  if (biasDims.size() != 1)
  {
    throw std::runtime_error("LinearGemmSubgraph::Convert(): cannot "
        "infer output size of ONNX Gemm operation!");
  }

  network.Add<mlpack::Linear>(biasDims[0]);
}

inline void LinearGemmSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the Gemm operation, and the biases must be the C matrix.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  int transB = 0;
  if (!ExtractAttribute(gemm, "transB", transB))
  {
    throw std::runtime_error("LinearGemmSubgraph::Convert(): cannot extract "
        "'transB' attribute!");
  }

  mlpack::Linear<>* l = dynamic_cast<mlpack::Linear<>*>(layers[0]);
  if (transB == 1)
    l->Weight() = TensorToArma(graph, gemm.input(1)).t();
  else
    l->Weight() = TensorToArma(graph, gemm.input(1));
  l->Bias() = TensorToArma(graph, gemm.input(2)).t();
}

} // namespace onnx_mlpack

#endif
