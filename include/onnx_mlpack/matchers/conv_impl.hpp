/**
 * @file conv_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_CONV_IMPL_HPP

#include "conv.hpp"
#include "../extract_attribute.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Conv layer.
 */
inline bool ConvSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MaxPool to ensure that we actually can
  // do the conversion.
  const onnx::NodeProto& conv = graph.node(nodes[0]);
  if (conv.op_type() != "Conv")
    return false;

  // There are a few restrictions that we have on a Conv node:
  //
  // 1. The kernel weights must be initialized.
  //
  // 2. `kernel_shape` must be two-dimensional.
  //
  // 3. `dilations` must be all 1 or not present.
  //
  // 4. `strides` must be 1 in any dimension higher than 2.
  //
  // 5. If `auto_pad` is SAME_UPPER or SAME_LOWER, we need to be able to
  //    determine the input tensor's size.
  //
  // 6. If explicit padding is given, padding in dimensions above the second
  //    must be zero.

  const std::string& wName = conv.input(1);
  bool foundInitializer = false;
  size_t maps = 0;
  size_t channels = 0;
  size_t kernelWidth = 0;
  size_t kernelHeight = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    // Kernel size is expected to be (M x (C/g) x H x W).
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == wName &&
        graph.initializer(i).dims_size() == 4)
    {
      foundInitializer = true;
      maps = graph.initializer(i).dims(0);
      channels = graph.initializer(i).dims(1);
      kernelHeight = graph.initializer(i).dims(2);
      kernelWidth = graph.initializer(i).dims(3);
    }
  }

  // If we didn't find the initializer and don't have a shape, we can't proceed.
  if (!foundInitializer)
    return false;

  // Make sure the kernel shape is two-dimensional.
  std::vector<int> kernelShape;
  if (!ExtractAttribute(conv, "kernel_shape", kernelShape))
    return false;
  if (kernelShape.size() == 0)
  {
    // Infer the kernel shape from the weights.
    kernelShape.push_back(kernelHeight);
    kernelShape.push_back(kernelWidth);
  }
  else if (kernelShape.size() != 2)
  {
    return false; // Kernel shape is invalid.
  }
  else if (foundInitializer &&
           (kernelShape[0] != kernelHeight || kernelShape[1] != kernelWidth))
  {
    return false; // Kernel shape does not match tensor.
  }
  else if (kernelShape[0] <= 0 || kernelShape[1] <= 0)
  {
    return false; // Invalid kernel shape.
  }

  // Check dilations.
  std::vector<int> dilations;
  if (!ExtractAttribute(conv, "dilations", dilations))
    return false;
  // Dilations must all be equal to 1.
  for (size_t i = 0; i < dilations.size(); ++i)
    if (dilations[i] != 1)
      return false;

  // Check strides.
  std::vector<int> strides;
  if (!ExtractAttribute(conv, "strides", strides))
    return false;
  if (strides.size() != 2)
    return false;
  if (strides[0] <= 0 || strides[1] <= 0)
    return false;

  // Check that we can infer the input shape if needed.
  std::string autoPad;
  if (!ExtractAttribute(conv, "auto_pad", autoPad))
    return false;
  // If auto_pad is SAME_UPPER or SAME_LOWER, then we need to be able to
  // determine the input tensor's size.
  if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
  {
    // Get the input width and height.
    size_t inputWidth = 0;
    size_t inputHeight = 0;
    for (size_t i = 0; i < graph.initializer_size(); ++i)
    {
      const onnx::TensorProto& t = graph.initializer(i);
      if (t.has_name() && t.name() == conv.input(0) && t.dims_size() == 4)
      {
        inputHeight = t.dims(2);
        inputWidth = t.dims(3);
        break;
      }
    }

    // If we did not find the input size, check the ValueInfoProtos too (so,
    // hopefully shape inference was successful).
    if (inputWidth == 0 && inputHeight == 0)
    {
      for (size_t i = 0; i < graph.value_info_size(); ++i)
      {
        const onnx::ValueInfoProto& v = graph.value_info(i);
        if (v.has_name() && v.name() == conv.input(0) && v.has_type() &&
            v.type().has_tensor_type() &&
            v.type().tensor_type().has_shape() &&
            v.type().tensor_type().shape().dim_size() == 4 &&
            v.type().tensor_type().shape().dim(2).has_dim_value() &&
            v.type().tensor_type().shape().dim(3).has_dim_value())
        {
          inputHeight = v.type().tensor_type().shape().dim(2).dim_value();
          inputWidth = v.type().tensor_type().shape().dim(3).dim_value();
          break;
        }
      }
    }

    // Make sure we could determine the shape of the input.
    if (inputWidth == 0 && inputHeight == 0)
      return false;
  }

  std::vector<int> pads;
  if (!ExtractAttribute(conv, "pads", pads))
    return false;
  // Explicit padding should be two-dimensional.
  if (pads.size() != 0 && pads.size() != 4)
    return false;

  // Get the number of groups, if it's grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "groups", groups))
    return false;

  // If we have a bias, ensure that it has the right size.
  if (conv.input_size() == 3)
  {
    const std::string& bName = conv.input(2);
    foundInitializer = false;
    for (size_t i = 0; i < graph.initializer_size(); ++i)
    {
      if (graph.initializer(i).has_name() &&
          graph.initializer(i).name() == bName &&
          graph.initializer(i).dims_size() >= 1)
      {
        if (graph.initializer(i).dims(0) != (maps / groups))
          return false;

        // All higher dimensions must be 1.
        for (size_t j = 1; j < graph.initializer(i).dims_size(); ++j)
          if (graph.initializer(i).dims(j) != 1)
            return false;
      }
    }
  }

  return true;
}

/**
 * Create a Convolution layer with the same metadata as the given ONNX graph.
 */
inline void ConvSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Given that the node is valid, we need to extract parameters from it.
  const onnx::NodeProto& conv = graph.node(nodes[0]);

  // We need to explicitly extract what the padding is, so that we can determine
  // if an extra Padding layer is going to be necessary (there are some cases in
  // which it is).
  std::vector<int> pads;
  if (!ExtractAttribute(conv, "pads", pads))
  {
    throw std::runtime_error("ConvSubgraph::Convert(): cannot extract 'pads' "
        "attribute!");
  }

  // Extract the strides.
  std::vector<int> strides;
  if (!ExtractAttribute(conv, "strides", strides))
  {
    throw std::runtime_error("ConvSubgraph::Convert(): cannot extract 'strides'"
        " attribute!");
  }
  if (strides.size() != 2)
    strides.resize(2, 1);

  // Extract the ceil_mode parameter.
  int ceilMode;
  if (!ExtractAttribute(conv, "ceil_mode", ceilMode))
  {
    throw std::runtime_error("ConvSubgraph::Convert(): cannot extract "
        "'ceil_mode' attribute!");
  }

  // Extract whether or not we are doing grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "groups", groups))
  {
    throw std::runtime_error("ConvSubgraph::Convert(): cannot extract 'groups'"
        " attribute!");
  }

  // Determine whether or not we are using a bias.
  bool useBias = (conv.input_size() == 3);

  // mlpack computes the number of channels as the third input dimension.
  // This will be implicit and does not need to be passed to the constructor of
  // the mlpack layers.
  const std::string& wName = conv.input(1);
  bool foundInitializer = false;
  size_t maps = 0;
  size_t channels = 0;
  size_t kernelWidth = 0;
  size_t kernelHeight = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == wName &&
        graph.initializer(i).dims_size() == 4)
    {
      foundInitializer = true;
      maps = graph.initializer(i).dims(0);
      channels = graph.initializer(i).dims(1);
      kernelHeight = graph.initializer(i).dims(2);
      kernelWidth = graph.initializer(i).dims(3);
    }
  }

  if (!foundInitializer)
  {
    throw std::runtime_error("ConvSubgraph::Convert(): cannot find kernel "
        "tensor '" + wName + "'!");
  }

  // Make sure the kernel shape is two-dimensional.
  std::vector<int> kernelShape;
  if (!ExtractAttribute(conv, "kernel_shape", kernelShape))
    return;
  if (kernelShape.size() == 0)
  {
    // Infer the kernel shape from the weights.
    kernelShape.push_back(kernelHeight);
    kernelShape.push_back(kernelWidth);
  }

  // Finally, compute the explicit padding values, if needed.
  if (pads.size() == 0)
  {
    std::string autoPad = "NOTSET";
    if (!ExtractAttribute(conv, "auto_pad", autoPad))
    {
      throw std::runtime_error("ConvSubgraph::Convert(): cannot extract "
          "'auto_pad' attribute!");
    }

    if (autoPad == "VALID")
    {
      pads.resize(4, 0); // No padding at all.
    }
    else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
    {
      // Force to two-dimensional.
      pads.resize(4, 0);

      // Get the input width and height.
      size_t inputWidth = 0;
      size_t inputHeight = 0;
      for (size_t i = 0; i < graph.initializer_size(); ++i)
      {
        const onnx::TensorProto& t = graph.initializer(i);
        if (t.has_name() && t.name() == conv.input(0) && t.dims_size() == 4)
        {
          inputHeight = t.dims(2);
          inputWidth = t.dims(3);
          break;
        }
      }

      // If we did not find the input size, check the ValueInfoProtos too (so,
      // hopefully shape inference was successful).
      if (inputWidth == 0 && inputHeight == 0)
      {
        for (size_t i = 0; i < graph.value_info_size(); ++i)
        {
          const onnx::ValueInfoProto& v = graph.value_info(i);
          if (v.has_name() && v.name() == conv.input(0) && v.has_type() &&
              v.type().has_tensor_type() &&
              v.type().tensor_type().has_shape() &&
              v.type().tensor_type().shape().dim_size() == 4 &&
              v.type().tensor_type().shape().dim(2).has_dim_value() &&
              v.type().tensor_type().shape().dim(3).has_dim_value())
          {
            inputHeight = v.type().tensor_type().shape().dim(2).dim_value();
            inputWidth = v.type().tensor_type().shape().dim(3).dim_value();
            break;
          }
        }
      }

      // Make sure we could determine the shape of the input.
      if (inputWidth == 0 && inputHeight == 0)
      {
        throw std::runtime_error("ConvSubgraph::Convert(): cannot determine "
            "shape of input tensor for SAME_UPPER/SAME_LOWER padding type!");
      }

      size_t totalPadHeight;
      size_t totalPadWidth;
      if (ceilMode == 0)
      {
        totalPadHeight = std::floor(double(inputWidth - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputWidth;
        totalPadWidth = std::floor(double(inputHeight - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputWidth;
      }
      else
      {
        totalPadHeight = std::ceil(double(inputWidth - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputWidth;
        totalPadWidth = std::ceil(double(inputHeight - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputWidth;
      }

      if (totalPadHeight % 2 == 0)
      {
        pads[0] = totalPadHeight / 2;
        pads[2] = totalPadHeight / 2;
      }
      else if (autoPad == "SAME_UPPER")
      {
        pads[0] = std::floor(totalPadHeight / 2.0);
        pads[2] = std::ceil(totalPadHeight / 2.0);
      }
      else
      {
        pads[0] = std::ceil(totalPadHeight / 2.0);
        pads[2] = std::floor(totalPadHeight / 2.0);
      }

      if (totalPadWidth % 2 == 0)
      {
        pads[1] = totalPadWidth / 2;
        pads[3] = totalPadWidth / 2;
      }
      else if (autoPad == "SAME_UPPER")
      {
        pads[1] = std::floor(totalPadWidth / 2.0);
        pads[3] = std::ceil(totalPadWidth / 2.0);
      }
      else
      {
        pads[1] = std::ceil(totalPadWidth / 2.0);
        pads[3] = std::floor(totalPadWidth / 2.0);
      }
    }
  }

  if (groups != 1)
  {
    network.Add<mlpack::GroupedConvolution>(
        maps, // output maps
        kernelShape[1], // width
        kernelShape[0], // height
        groups,
        strides[1], // stride width
        strides[0], // stride height
        std::make_tuple(size_t(pads[1]), size_t(pads[3])),
        std::make_tuple(size_t(pads[0]), size_t(pads[2])),
        "none", // use explicit padding values
        useBias);
  }
  else
  {
    network.Add<mlpack::Convolution>(
        maps, // output maps
        kernelShape[1], // width
        kernelShape[0], // height
        strides[1], // stride width
        strides[0], // stride height
        std::make_tuple(size_t(pads[1]), size_t(pads[3])),
        std::make_tuple(size_t(pads[0]), size_t(pads[2])),
        "none", // use explicit padding values
        useBias);
  }
}

/**
 * Convert the weights for a matching of the Conv layer.
 */
inline void ConvSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  const onnx::NodeProto& conv = graph.node(nodes[0]);
  const bool hasBias = (conv.input_size() == 3);
  const std::string& kName = conv.input(1);

  size_t kIndex = graph.initializer_size();
  size_t bIndex = graph.initializer_size();
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == kName && t.dims_size() == 4)
      kIndex = i;

    if (hasBias && t.has_name() && t.name() == conv.input(2))
      bIndex = i;
  }

  if (kIndex == graph.initializer_size())
  {
    throw std::runtime_error("ConvSubgraph::TransferWeights(): could not find "
        "suitable kernel tensor '" + kName + "'!");
  }

  if (hasBias && bIndex == graph.initializer_size())
  {
    throw std::runtime_error("ConvSubgraph::TransferWeights(): could not find "
        "suitable bias tensor '" + conv.input(2) + "'!");
  }

  arma::mat bias, kernel;
  kernel = TensorToArma(graph.initializer(kIndex), true);
  if (hasBias)
    bias = TensorToArma(graph.initializer(bIndex), true);

  // The input weights are stored in tensor shape (M x C x H x W), but our
  // internal representation is flipped and will be (W x H x C x M).  This is,
  // however, the same vectorized representation!

  // Extract whether or not we are doing grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "groups", groups))
  {
    throw std::runtime_error("ConvSubgraph::TransferWeights(): cannot extract "
        "'groups' attribute!");
  }

  if (groups == 1)
  {
    mlpack::Convolution<>* l = dynamic_cast<mlpack::Convolution<>*>(layers[0]);

    // Expected size of bias: maps x 1.
    if (hasBias)
      l->Bias() = bias;
    else
      l->Bias().zeros();

    // Expected size of weight tensor: maps * kernelWidth *
    // kernelHeight.  Use an alias to reshape the weights correctly.
    l->Weight() = arma::cube(kernel.memptr(), l->Weight().n_rows,
        l->Weight().n_cols, l->Weight().n_slices, true, false);
  }
  else
  {
    mlpack::GroupedConvolution<>* l =
        dynamic_cast<mlpack::GroupedConvolution<>*>(layers[0]);

    // Expected size of bias: maps x 1.
    if (hasBias)
      l->Bias() = bias;
    else
      l->Bias().zeros();

    // Expected size of weight tensor: maps * kernelWidth *
    // kernelHeight.  Use an alias to reshape the weights correctly.
    l->Weight() = arma::cube(kernel.memptr(), l->Weight().n_rows,
        l->Weight().n_cols, l->Weight().n_slices, true, false);
  }
}

} // namespace onnx_mlpack

#endif
