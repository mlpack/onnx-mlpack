/**
 * @file max_pooling_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the MaxPooling layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MAX_POOLING_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MAX_POOLING_IMPL_HPP

#include "max_pooling.hpp"
#include "../extract_attribute.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a MaxPooling layer.
 */
inline bool MaxPoolingSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MaxPool to ensure that we actually can
  // do the conversion.
  const onnx::NodeProto& maxPool = graph.node(nodes[0]);
  if (maxPool.op_type() != "MaxPool")
    return false;

  // The ONNX MaxPool node supports a number of different arguments, but not all
  // of them are supported by mlpack.  Also, the input shape to the MaxPool
  // operation must be 2-dimensional.
  //
  // auto_pad: { "NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID" }
  //   - (this option is deprecated but may still be present)
  //   - "NOTSET": 'pads' is used
  //   - "SAME_UPPER"/"SAME_UPPER": a formula can be used to compute padding
  //   - "VALID": no padding is needed
  //
  // ceil_mode: 0/1
  //   This parameter is supported by mlpack (it's the "floor" parameter
  //   inverted).
  //
  // dilations: [int...]
  //   mlpack does not support dilations in a MaxPooling layer, so, this either
  //   needs to be not present, or all values need to be 1.
  //
  // kernel_shape: [int...]
  //   So long as this is 2-dimensional, these are the kernelWidth and
  //   kernelHeight parameters.
  //
  // pads: [int...]
  //   mlpack's MaxPooling layer only supports 'valid' padding.  So, if this is
  //   present, then we need to add an extra padding layer.
  //
  // storage_order: int
  //   This is only relevant when the 'indices' output is used.
  //
  // strides: [int...]
  //   So long as this is 2-dimensional, these are the strideWidth and
  //   strideHeight parameters.

  // To work with this particular rule, we have a handful of restrictions:
  //
  // 1. kernel_shape and strides are either not present or two-dimensional.
  //
  // 2. pads must be all zeros in any dimensions higher than 2.
  //
  // 3. dilations must either be not present or all values must be 1.
  //
  // 4. The optional indices output must not be used by any other node in the
  //    graph.

  // First, look through the attributes to check whether the first three
  // restrictions are satisfied.
  std::string autoPad;
  if (!ExtractAttribute(maxPool, "auto_pad", autoPad))
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
      if (t.has_name() && t.name() == maxPool.input(0) && t.dims_size() >= 2)
      {
        inputWidth = t.dims(0);
        inputHeight = t.dims(1);
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
        if (v.has_name() && v.name() == maxPool.input(0) && v.has_type() &&
            v.type().has_tensor_type() &&
            v.type().tensor_type().has_shape() &&
            v.type().tensor_type().shape().dim_size() >= 2 &&
            v.type().tensor_type().shape().dim(0).has_dim_value() &&
            v.type().tensor_type().shape().dim(1).has_dim_value())
        {
          inputWidth = v.type().tensor_type().shape().dim(0).dim_value();
          inputHeight = v.type().tensor_type().shape().dim(1).dim_value();
          break;
        }
      }
    }

    // Make sure we could determine the shape of the input.
    if (inputWidth == 0 && inputHeight == 0)
      return false;
  }

  int ceilMode;
  if (!ExtractAttribute(maxPool, "ceil_mode", ceilMode))
    return false;
  if (ceilMode != 0 && ceilMode != 1)
    return false;

  std::vector<int> dilations;
  if (!ExtractAttribute(maxPool, "dilations", dilations))
    return false;
  // Dilations must all be equal to 1.
  for (size_t i = 0; i < dilations.size(); ++i)
    if (dilations[i] != 1)
      return false;

  std::vector<int> kernelShape;
  if (!ExtractAttribute(maxPool, "kernel_shape", kernelShape))
    return false;
  // The kernel shape must be two-dimensional.
  if (kernelShape.size() != 2)
    return false;
  if (kernelShape[0] <= 0 || kernelShape[1] <= 0)
    return false;

  std::vector<int> pads;
  if (!ExtractAttribute(maxPool, "pads", pads))
    return false;
  // Dimensions after the second must be zero padding.
  const size_t padDims = pads.size() / 2;
  for (size_t j = 2; j < padDims; ++j)
  {
    if (pads[j] != 0)
      return false;
    if (pads[padDims + j] != 0)
      return false;
  }

  std::vector<int> strides;
  if (!ExtractAttribute(maxPool, "strides", strides))
    return false;
  if (strides.size() != 2)
    return false;
  if (strides[0] <= 0 || strides[1] <= 0)
    return false;

  // Next, make sure the second output (if it exists) is not used anywhere else
  // in the graph.
  if (maxPool.output_size() == 2)
  {
    // Iterate over all nodes in the graph to see if this output is used.
    const std::string& indicesName = maxPool.output(1);
    for (size_t i = 0; i < graph.node_size(); ++i)
    {
      const onnx::NodeProto& n = graph.node(i);
      for (size_t j = 0; j < n.input_size(); ++j)
      {
        if (n.input(j) == indicesName)
          return false; // The output is used!
      }
    }
  }

  return true;
}

/**
 * Create a MaxPooling layer with the same metadata as the given ONNX graph.
 */
inline void MaxPoolingSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Given that the node is valid, we need to extract parameters from it.
  const onnx::NodeProto& maxPool = graph.node(nodes[0]);

  // We need to add an extra preliminary padding layer if the ONNX node's
  // padding is not "VALID", or if the padding vector is not all zeroes.
  std::string autoPad = "NOTSET";
  if (!ExtractAttribute(maxPool, "auto_pad", autoPad))
  {
    throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot extract "
        "'auto_pad' attribute!");
  }

  int ceilMode = 0;
  if (!ExtractAttribute(maxPool, "ceil_mode", ceilMode))
  {
    throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot extract "
        "'ceil_mode' attribute!");
  }

  std::vector<int> kernelShape;
  if (!ExtractAttribute(maxPool, "kernel_shape", kernelShape))
  {
    throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot extract "
        "'kernel_shape' attribute!");
  }

  std::vector<int> pads;
  if (!ExtractAttribute(maxPool, "pads", pads))
  {
    throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot extract "
        "'pads' attribute!");
  }

  std::vector<int> strides;
  if (!ExtractAttribute(maxPool, "strides", strides))
  {
    throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot extract "
        "'strides' attribute!");
  }

  // We might need to add a padding layer too, if the ONNX node's padding is not
  // "VALID".
  bool allPadsZero = true;
  for (size_t i = 0; i < pads.size(); ++i)
  {
    if (pads[i] != 0)
    {
      allPadsZero = false;
      break;
    }
  }

  if (autoPad != "VALID" || !allPadsZero)
  {
    // If pads is not set, manually set it.
    if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
    {
      // Force to two-dimensional.
      pads.resize(4);

      // Get the input width and height.
      size_t inputWidth = 0;
      size_t inputHeight = 0;
      for (size_t i = 0; i < graph.initializer_size(); ++i)
      {
        const onnx::TensorProto& t = graph.initializer(i);
        if (t.has_name() && t.name() == maxPool.input(0) && t.dims_size() >= 2)
        {
          inputWidth = t.dims(0);
          inputHeight = t.dims(1);
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
          if (v.has_name() && v.name() == maxPool.input(0) && v.has_type() &&
              v.type().has_tensor_type() &&
              v.type().tensor_type().has_shape() &&
              v.type().tensor_type().shape().dim_size() >= 2 &&
              v.type().tensor_type().shape().dim(0).has_dim_value() &&
              v.type().tensor_type().shape().dim(1).has_dim_value())
          {
            inputWidth = v.type().tensor_type().shape().dim(0).dim_value();
            inputHeight = v.type().tensor_type().shape().dim(1).dim_value();
            break;
          }
        }
      }

      // Make sure we could determine the shape of the input.
      if (inputWidth == 0 && inputHeight == 0)
      {
        throw std::runtime_error("MaxPoolingSubgraph::Convert(): cannot "
            "determine shape of input tensor for SAME_UPPER/SAME_LOWER padding "
            "type!");
      }

      size_t totalPadWidth;
      size_t totalPadHeight;
      if (ceilMode == 0)
      {
        totalPadWidth = std::floor(double(inputWidth - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputWidth;
        totalPadHeight = std::floor(double(inputHeight - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputWidth;
      }
      else
      {
        totalPadWidth = std::ceil(double(inputWidth - 1) / strides[0]) *
            strides[0] + kernelShape[0] - inputWidth;
        totalPadHeight = std::ceil(double(inputHeight - 1) / strides[1]) *
            strides[1] + kernelShape[1] - inputWidth;
      }

      if (totalPadWidth % 2 == 0)
      {
        pads[0] = totalPadWidth / 2;
        pads[2] = totalPadWidth / 2;
      }
      else if (autoPad == "SAME_UPPER")
      {
        pads[0] = std::floor(totalPadWidth / 2.0);
        pads[2] = std::ceil(totalPadWidth / 2.0);
      }
      else
      {
        pads[0] = std::ceil(totalPadWidth / 2.0);
        pads[2] = std::floor(totalPadWidth / 2.0);
      }

      if (totalPadHeight % 2 == 0)
      {
        pads[1] = totalPadHeight / 2;
        pads[3] = totalPadHeight / 2;
      }
      else if (autoPad == "SAME_UPPER")
      {
        pads[1] = std::floor(totalPadHeight / 2.0);
        pads[3] = std::ceil(totalPadHeight / 2.0);
      }
      else
      {
        pads[1] = std::ceil(totalPadHeight / 2.0);
        pads[3] = std::floor(totalPadHeight / 2.0);
      }
    }

    const size_t padDims = pads.size() / 2;

    network.Add<mlpack::Padding>(pads[0], // left
                                 pads[padDims], // right
                                 pads[1], // top
                                 pads[padDims + 1]); // bottom
  }

  // Set strides to [1, 1] if not set.
  if (strides.size() == 0)
    strides.resize(2, 1);

  network.Add<mlpack::MaxPooling>(kernelShape[0], // kernel width
                                  kernelShape[1], // kernel height
                                  strides[0], // stride width
                                  strides[1], // stride height
                                  (ceilMode == 0)); // floor
}

} // namespace onnx_mlpack

#endif
