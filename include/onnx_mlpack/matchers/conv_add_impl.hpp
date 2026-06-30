/**
 * @file conv_add_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_ADD_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_CONV_ADD_IMPL_HPP

#include "conv_add.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Conv layer.
 */
inline bool ConvAddSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 2)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;
  if (nodes[1] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MaxPool to ensure that we actually can
  // do the conversion.
  const onnx::NodeProto& conv = graph.node(nodes[0]);
  if (conv.op_type() != "Conv")
    return false;
  const onnx::NodeProto& add = graph.node(nodes[1]);
  if (add.op_type() != "Add")
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
  std::vector<size_t> kernelDims;
  ExtractTensorDims(graph, wName, kernelDims, true);
  if (kernelDims.size() != 4)
    return false;

  // Make sure the kernel shape is two-dimensional.
  std::vector<int> kernelShape;
  if (!ExtractAttribute(conv, "kernel_shape", kernelShape))
    return false;
  if (kernelShape.size() == 0)
  {
    // Infer the kernel shape from the weights.
    kernelShape.push_back(kernelDims[2]);
    kernelShape.push_back(kernelDims[3]);
  }
  else if (kernelShape.size() != 2)
  {
    return false; // Kernel shape is invalid.
  }
  else if (kernelShape[0] != kernelDims[2] || kernelShape[1] != kernelDims[3])
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
    std::vector<size_t> inputDims;
    ExtractTensorDims(graph, conv.input(0), inputDims);
    if (inputDims.size() != 4)
      return false;

    // Make sure we could determine the shape of the input.
    if (inputDims[2] == 0 && inputDims[3] == 0)
      return false;
  }

  std::vector<int> pads;
  if (!ExtractAttribute(conv, "pads", pads))
    return false;
  // Explicit padding should be two-dimensional.
  if (pads.size() != 0 && pads.size() != 4)
    return false;

  // The conv node needs to have no bias.
  if (conv.input_size() == 3)
    return false;

  // Get the number of groups, if it's grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "group", groups))
    return false;

  // The bias in the add node must have the right shape.
  // Note that either input of the add node could be the bias.
  const size_t bIndex = (conv.output(0) == add.input(0) ? 1 : 0);
  const std::string& bName = add.input(bIndex);
  std::vector<size_t> biasDims;
  ExtractTensorDims(graph, add.input(bIndex), biasDims, true);
  if (biasDims.size() == 0)
    return false;
  if (biasDims[0] != (kernelDims[0] / groups))
    return false;
  // All higher dimensions must be 1.
  for (size_t j = 1; j < biasDims.size(); ++j)
    if (biasDims[j] != 1)
      return false;

  return true;
}

/**
 * Create a Convolution layer with the same metadata as the given ONNX graph.
 */
inline void ConvAddSubgraph::Convert(
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
    throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract 'pads'"
        " attribute!");
  }

  // Extract the strides.
  std::vector<int> strides;
  if (!ExtractAttribute(conv, "strides", strides))
  {
    throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract "
        "'strides' attribute!");
  }
  if (strides.size() != 2)
    strides.resize(2, 1);

  // Extract the ceil_mode parameter.
  int ceilMode;
  if (!ExtractAttribute(conv, "ceil_mode", ceilMode))
  {
    throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract "
        "'ceil_mode' attribute!");
  }

  // Extract whether or not we are doing grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "group", groups))
  {
    throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract "
        "'groups' attribute!");
  }

  // mlpack computes the number of channels as the third input dimension.
  // This will be implicit and does not need to be passed to the constructor of
  // the mlpack layers.
  const std::string& wName = conv.input(1);
  std::vector<size_t> kernelDims;
  ExtractTensorDims(graph, conv.input(1), kernelDims, true);
  if (kernelDims.size() != 4)
  {
    throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract size "
        "of kernel tensor!");
  }

  // Make sure the kernel shape is two-dimensional.
  std::vector<int> kernelShape;
  if (!ExtractAttribute(conv, "kernel_shape", kernelShape))
    return;
  if (kernelShape.size() == 0)
  {
    // Infer the kernel shape from the weights.
    kernelShape.push_back(kernelDims[2]);
    kernelShape.push_back(kernelDims[3]);
  }
  if (kernelShape.size() != 2 || kernelShape[0] <= 0 || kernelShape[1] <= 0)
  {
    throw std::runtime_error("ConvAddSubgraph::Convert(): got invalid kernel "
        "shape!");
  }

  // Finally, compute the explicit padding values, if needed.
  if (pads.size() == 0)
  {
    std::string autoPad = "NOTSET";
    if (!ExtractAttribute(conv, "auto_pad", autoPad))
    {
      throw std::runtime_error("ConvAddSubgraph::Convert(): cannot extract "
          "'auto_pad' attribute!");
    }

    if (autoPad == "VALID" || autoPad == "NOTSET")
    {
      pads.resize(4, 0); // No padding at all.
    }
    else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
    {
      // Force to four-dimensional.
      pads.resize(4, 0);

      // Get the input width and height.
      std::vector<size_t> inputDims;
      ExtractTensorDims(graph, conv.input(0), inputDims);

      // Make sure we could determine the shape of the input.
      if (inputDims[2] == 0 && inputDims[3] == 0)
      {
        throw std::runtime_error("ConvAddSubgraph::Convert(): cannot determine "
            "shape of input tensor for SAME_UPPER/SAME_LOWER padding type!");
      }

      size_t totalPadHeight;
      size_t totalPadWidth;
      if (ceilMode == 0)
      {
        totalPadHeight = std::floor(double(inputDims[2] - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputDims[2];
        totalPadWidth = std::floor(double(inputDims[3] - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputDims[3];
      }
      else
      {
        totalPadHeight = std::ceil(double(inputDims[2] - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputDims[2];
        totalPadWidth = std::ceil(double(inputDims[3] - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputDims[3];
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
        kernelDims[0], // output maps
        kernelShape[1], // width
        kernelShape[0], // height
        groups,
        strides[1], // stride width
        strides[0], // stride height
        std::make_tuple(size_t(pads[1]), size_t(pads[3])),
        std::make_tuple(size_t(pads[0]), size_t(pads[2])),
        "none", // use explicit padding values
        true); // use bias
  }
  else
  {
    network.Add<mlpack::Convolution>(
        kernelDims[0], // output maps
        kernelShape[1], // width
        kernelShape[0], // height
        strides[1], // stride width
        strides[0], // stride height
        std::make_tuple(size_t(pads[1]), size_t(pads[3])),
        std::make_tuple(size_t(pads[0]), size_t(pads[2])),
        "none", // use explicit padding values
        true); // use bias
  }
}

/**
 * Convert the weights for a matching of the Conv layer.
 */
inline void ConvAddSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    std::vector<mlpack::Layer<>*>& layers) const
{
  const onnx::NodeProto& conv = graph.node(nodes[0]);
  const onnx::NodeProto& add = graph.node(nodes[1]);

  // Note that either input of the add node could be the bias.
  const size_t bInputIndex = (conv.output(0) == add.input(0) ? 1 : 0);

  arma::mat bias, kernel;
  kernel = TensorToArma(graph, conv.input(1), true);
  bias = TensorToArma(graph, add.input(bInputIndex), true);

  // The input weights are stored in tensor shape (M x C x H x W), but our
  // internal representation is flipped and will be (W x H x C x M).  This is,
  // however, the same vectorized representation!

  // Extract whether or not we are doing grouped convolution.
  int groups = 1;
  if (!ExtractAttribute(conv, "group", groups))
  {
    throw std::runtime_error("ConvSubgraph::TransferWeights(): cannot extract "
        "'groups' attribute!");
  }

  if (groups == 1)
  {
    mlpack::Convolution<>* l = dynamic_cast<mlpack::Convolution<>*>(layers[0]);

    // Expected size of bias: maps x 1.
    l->Bias() = bias;

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
    l->Bias() = bias;

    // Expected size of weight tensor: maps * kernelWidth *
    // kernelHeight.  Use an alias to reshape the weights correctly.
    l->Weight() = arma::cube(kernel.memptr(), l->Weight().n_rows,
        l->Weight().n_cols, l->Weight().n_slices, true, false);
  }
}

} // namespace onnx_mlpack

#endif
