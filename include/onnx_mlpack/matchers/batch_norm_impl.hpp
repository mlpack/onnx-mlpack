/**
 * @file batch_norm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the BatchNorm layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_BATCH_NORM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_BATCH_NORM_IMPL_HPP

#include "linear_no_bias_gemm.hpp"
#include "../tensor_to_arma.hpp"

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
  const std::string& scaleName = bn.input(1);
  const std::string& biasName = bn.input(2);
  const std::string& meanName = bn.input(3);
  const std::string& varName = bn.input(4);
  size_t scaleDims = 0;
  size_t biasDims = 0;
  size_t meanDims = 0;
  size_t varDims = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == scaleName &&
        graph.initializer(i).dims_size() >= 1)
    {
      scaleDims = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        scaleDims *= graph.initializer(i).dims(j);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == biasName &&
        graph.initializer(i).dims_size() >= 1)
    {
      biasDims = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        biasDims *= graph.initializer(i).dims(j);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == meanName &&
        graph.initializer(i).dims_size() >= 1)
    {
      meanDims = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        meanDims *= graph.initializer(i).dims(j);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == meanName &&
        graph.initializer(i).dims_size() >= 1)
    {
      varDims = graph.initializer(i).dims(0);
      for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
        varDims *= graph.initializer(i).dims(j);
    }
  }

  // Now we have to extract the shape of the input tensor, to make sure that the
  // shapes of all the weights match.
  size_t channels = 0;
  const std::string& inputName = bn.input(0);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == inputName && t.dims_size() >= 3)
    {
      channels = t.dims(1);
      break;
    }

    // If we did not find the input size, check the ValueInfoProtos too (so,
    // hopefully shape inference was successful).
    if (channels == 0)
    {
      for (size_t i = 0; i < graph.value_info_size(); ++i)
      {
        const onnx::ValueInfoProto& v = graph.value_info(i);
        if (v.has_name() && v.name() == inputName && v.has_type() &&
            v.type().has_tensor_type() &&
            v.type().tensor_type().has_shape() &&
            v.type().tensor_type().shape().dim_size() >= 3 &&
            v.type().tensor_type().shape().dim(1).has_dim_value())
        {
          channels = v.type().tensor_type().shape().dim(1).dim_value();
          break;
        }
      }
    }
  }

  // Make sure the dimensions match.
  if (scaleDims != channels)
    return false;
  if (biasDims != channels)
    return false;
  if (meanDims != channels)
    return false;
  if (varDims != channels)
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
  size_t inputDims = 0;
  const std::string& inputName = bn.input(0);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == inputName && t.dims_size() >= 3)
    {
      inputDims = t.dims_size();
      break;
    }

    // If we did not find the input size, check the ValueInfoProtos too (so,
    // hopefully shape inference was successful).
    if (inputDims == 0)
    {
      for (size_t i = 0; i < graph.value_info_size(); ++i)
      {
        const onnx::ValueInfoProto& v = graph.value_info(i);
        if (v.has_name() && v.name() == inputName && v.has_type() &&
            v.type().has_tensor_type() &&
            v.type().tensor_type().has_shape() &&
            v.type().tensor_type().shape().dim_size() >= 3)
        {
          inputDims = v.type().tensor_type().shape().dim_size();
          break;
        }
      }
    }
  }

  if (inputDims == 0)
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
  network.Add<mlpack::BatchNorm>(inputDims - 2, inputDims - 2, double(epsilon),
      false /* ONNX always uses momentum */, double(momentum));
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
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bn.input(1))
    {
      scale = TensorToArma(graph.initializer(i), true);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bn.input(2))
    {
      bias = TensorToArma(graph.initializer(i), true);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bn.input(3))
    {
      mean = TensorToArma(graph.initializer(i), true);
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bn.input(4))
    {
      var = TensorToArma(graph.initializer(i), true);
    }
  }

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
