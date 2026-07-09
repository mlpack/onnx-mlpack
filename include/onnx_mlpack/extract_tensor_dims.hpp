/**
 * @file extract_tensor_dims.hpp
 * @author Ryan Curtin
 *
 * Given the name of a tensor, extract its dimensions and return a bool
 * indicating whether the operation was successful.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_EXTRACT_TENSOR_DIMS_HPP
#define ONNX_MLPACK_EXTRACT_TENSOR_DIMS_HPP

#include <armadillo>
#include <onnx/onnx_pb.h>

namespace onnx_mlpack {

/**
 * Extract the dimensions of a tensor as a std::vector<size_t>.  If extraction
 * fails, then dims.size() will be set to 0.  Any -1/named dimension sizes will
 * be set to 0.
 */
template<typename eT = double>
void ExtractTensorDims(const onnx::GraphProto& graph,
                       const std::string& tensorName,
                       std::vector<size_t>& dims,
                       const bool initializersOnly = false)
{
  // Try to find the tensor in the initializers so we can extract its size.
  dims.clear();
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == tensorName)
    {
      for (size_t d = 0; d < t.dims_size(); ++d)
        dims.push_back(t.dims(d));
      break;
    }
  }

  // If we didn't get anything, try looking in the ValueInfoProtos, if we are
  // allwoed to.
  if (dims.size() == 0 && !initializersOnly)
  {
    for (size_t i = 0; i < graph.value_info_size(); ++i)
    {
      const onnx::ValueInfoProto& v = graph.value_info(i);
      if (v.has_name() && v.name() == tensorName && v.has_type() &&
          v.type().has_tensor_type() && v.type().tensor_type().has_shape())
      {
        for (size_t d = 0; d < v.type().tensor_type().shape().dim_size(); ++d)
        {
          if (v.type().tensor_type().shape().dim(d).has_dim_value())
            dims.push_back(v.type().tensor_type().shape().dim(d).dim_value());
          else
            dims.push_back(0); // symbolic dimensions get represented as 0
        }
      }
    }
  }
}

} // namespace onnx_mlpack

#endif
