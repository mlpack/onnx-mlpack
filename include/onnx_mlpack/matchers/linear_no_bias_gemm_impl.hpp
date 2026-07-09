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
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_IMPL_HPP

#include "linear_no_bias_gemm.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool LinearNoBiasGemmSubgraph::Validate(
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
  if (gemm.input_size() != 2 && gemm.input_size() != 3)
    return false;

  // We require that the second input parameter, the weights, are fully
  // initialized.
  std::vector<size_t> weightDims;
  ExtractTensorDims(graph, gemm.input(1), weightDims, true);
  if (weightDims.size() != 2)
    return false;

  // We cannot accept when beta is not 0 and C is specified.  (beta > 0 implies
  // recurrence!)
  if (gemm.input_size() == 3)
  {
    float beta = 1.0f;
    if (!ExtractAttribute(gemm, "beta", beta))
      return false;

    if (beta != 0.0f)
      return false;
  }

  // The alpha attribute must be set to 1; the LinearNoBias layer doesn't
  // support constant scaling.
  float alpha = 1.0f;
  if (!ExtractAttribute(gemm, "alpha", alpha))
    return false;

  if (alpha != 1.0f)
    return false;

  return true;
}

/**
 * Create a LinearNoBias layer with the same metadata as the given ONNX graph.
 */
inline void LinearNoBiasGemmSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a Gemm, we don't have anything to do
  // other than create the LinearNoBias layer.  However, we must first compute
  // the number of output nodes using the shape of the graph.
  //
  // There are a few possibilities: if C is specified, we can steal the size
  // from there.  If C is not specified, then we must infer the size based on
  // the shapes of A and B (and the settings of transA and transB).

  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  size_t outputDims = 0;
  std::vector<size_t> weightDims;
  ExtractTensorDims(graph, gemm.input(1), weightDims);

  int transB = 0;
  if (!ExtractAttribute(gemm, "transB", transB))
  {
    throw std::runtime_error("LinearNoBiasGemmSubgraph::Convert(): could not "
        "extract 'transB' attribute from Gemm operator!");
  }

  outputDims = (transB == 0) ? weightDims[1] : weightDims[0];

  if (outputDims == 0)
  {
    throw std::runtime_error("LinearNoBiasGemmSubgraph::Convert(): cannot "
        "infer output size of ONNX Gemm operation!");
  }

  network.Add<mlpack::LinearNoBias>(outputDims);
}

inline void LinearNoBiasGemmSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the Gemm operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  const std::string bName = gemm.input(1);

  int transB = 0;
  if (!ExtractAttribute(gemm, "transB", transB))
  {
    throw std::runtime_error("LinearNoBiasSubgraph::TransferWeights(): cannot "
        "extract 'transB' attribute from Gemm operator!");
  }

  mlpack::LinearNoBias<>* l = dynamic_cast<mlpack::LinearNoBias<>*>(layers[0]);
  if (transB == 1)
    l->Parameters() = TensorToArma(graph, gemm.input(1)).t();
  else
    l->Parameters() = TensorToArma(graph, gemm.input(1));
}

} // namespace onnx_mlpack

#endif
