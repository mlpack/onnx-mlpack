/**
 * @file batch_norm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the BatchNorm layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_BATCH_NORM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_BATCH_NORM_IMPL_HPP

#include "batch_norm.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a BatchNorm layer.
 */
inline bool BatchNormSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the gemm to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& bn = graph.node(nodes[0]);
  if (bn.op_type() != "BatchNormalization")
    return false;
  if (bn.input_size() != 5)
    return false;

  // We can only match the node if the optional outputs (running_mean and
  // running_var) aren't actually used anywhere.
  if (bn.output_size() != 1)
    return false;

  // We require that the four weights (scale/bias/running_mean/running_variance)
  // are all initialized.
  std::vector<size_t> scaleDims, biasDims, meanDims, varDims;
  ExtractTensorDims(graph, bn.input(1), scaleDims, true);
  ExtractTensorDims(graph, bn.input(2), biasDims, true);
  ExtractTensorDims(graph, bn.input(3), meanDims, true);
  ExtractTensorDims(graph, bn.input(4), varDims, true);

  if (scaleDims.size() == 0)
    return false;
  if (biasDims.size() == 0)
    return false;
  if (meanDims.size() == 0)
    return false;
  if (varDims.size() == 0)
    return false;

  size_t totalScaleDims = scaleDims[0];
  for (size_t j = 1; j < scaleDims.size(); ++j)
    totalScaleDims *= scaleDims[j];

  size_t totalBiasDims = biasDims[0];
  for (size_t j = 1; j < biasDims.size(); ++j)
    totalBiasDims *= biasDims[j];

  size_t totalMeanDims = meanDims[0];
  for (size_t j = 1; j < meanDims.size(); ++j)
    totalMeanDims *= meanDims[j];

  size_t totalVarDims = varDims[0];
  for (size_t j = 1; j < varDims.size(); ++j)
    totalVarDims *= varDims[j];

  std::vector<size_t> inputDims;
  ExtractTensorDims(graph, bn.input(0), inputDims);
  if (inputDims.size() != 4)
    return false;

  // Make sure the dimensions match.  inputDims[1] is the number of channels.
  if (totalScaleDims != inputDims[1])
    return false;
  if (totalBiasDims != inputDims[1])
    return false;
  if (totalMeanDims != inputDims[1])
    return false;
  if (totalVarDims != inputDims[1])
    return false;

  return true;
}

/**
 * Create a BatchNorm layer with the same metadata as the given ONNX graph.
 */
inline void BatchNormSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  const onnx::NodeProto& bn = graph.node(nodes[0]);

  // To create the mlpack BatchNorm layer, we need to know only two things:
  //
  //  * The number of dimensions we are applying BatchNorm to.
  //  * The epsilon and momentum parameters.

  // For the first, we can extract the number of dimensions from the input
  // tensor shape.
  std::vector<size_t> inputDims;
  ExtractTensorDims(graph, bn.input(0), inputDims);
  if (inputDims.size() == 0)
  {
    throw std::runtime_error("BatchNormSubgraph::Convert(): could not extract "
        "the number of dimensions in the input tensor!");
  }

  // Extract the eps and momentum attributes.
  float epsilon = 1e-5f;
  if (!ExtractAttribute(bn, "epsilon", epsilon))
  {
    throw std::runtime_error("BatchNormSubgraph::Convert(): could not extract "
        "'epsilon' attribute from BatchNormalization operator!");
  }
  float momentum = 0.9f;
  if (!ExtractAttribute(bn, "momentum", momentum))
  {
    throw std::runtime_error("BatchNormSubgraph::Convert(): could not extract "
        "'momentum' attribute from BatchNormalization operator!");
  }

  // Batch normalization is only applied on the channel axis, which will end up
  // being the last axis.
  network.Add<mlpack::BatchNorm>(inputDims.size() - 2, inputDims.size() - 2,
      double(epsilon), false /* ONNX always uses momentum */, double(momentum));
}

inline void BatchNormSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  // We need to extract the scale, bias, mean, and variance and set the elements
  // accordingly in the layer.
  const onnx::NodeProto& bn = graph.node(nodes[0]);

  // We can just extract the individual tensors as vectors, since that is how
  // mlpack shapes them.
  arma::mat scale, bias, mean, var;
  scale = TensorToArma(graph, bn.input(1), true);
  bias = TensorToArma(graph, bn.input(2), true);
  mean = TensorToArma(graph, bn.input(3), true);
  var = TensorToArma(graph, bn.input(4), true);

  // Make sure we were able to extract all the vectors.
  if (scale.n_elem == 0)
  {
    throw std::runtime_error("BatchNormSubgraph::TransferWeights(): could not "
        "extract scale tensor!");
  }

  if (bias.n_elem == 0)
  {
    throw std::runtime_error("BatchNormSubgraph::TransferWeights(): could not "
        "extract bias tensor!");
  }

  if (mean.n_elem == 0)
  {
    throw std::runtime_error("BatchNormSubgraph::TransferWeights(): could not "
        "extract input_mean tensor!");
  }

  if (var.n_elem == 0)
  {
    throw std::runtime_error("BatchNormSubgraph::TransferWeights(): could not "
        "extract input_var tensor!");
  }

  mlpack::BatchNorm<>* l = dynamic_cast<mlpack::BatchNorm<>*>(layers[0]);

  l->Gamma() = scale;
  l->Beta() = bias;
  l->TrainingMean() = mean;
  l->TrainingVariance() = var;
}

} // namespace onnx_mlpack

#endif
