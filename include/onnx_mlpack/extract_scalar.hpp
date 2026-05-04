/**
 * @file extract_scalar.hpp
 * @author Ryan Curtin
 *
 * Given the name of a tensor, extract a scalar and return a bool indicating
 * whether the operation was successful.
 */
#ifndef ONNX_MLPACK_EXTRACT_SCALAR_HPP
#define ONNX_MLPACK_EXTRACT_SCALAR_HPP

#include <armadillo>
#include <onnx/onnx_pb.h>

namespace onnx_mlpack {

template<typename eT = double>
bool ExtractScalar(const onnx::GraphProto& graph,
                   const std::string& tensorName,
                   eT& val)
{
  // Try to find the tensor in the initializers so we can extract its size.
  size_t dims = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == tensorName)
    {
      // Data must not be externally stored.
      if (t.external_data_size() > 0)
        return false;

      // If the data is more than one element, we have a problem.
      const bool raw = t.has_raw_data();
      const size_t rs = raw ? t.raw_data().size() : 0;
      switch (t.data_type())
      {
        case onnx::TensorProto::FLOAT:
          {
          const size_t s = raw ? (rs / 4) : t.float_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((float*) t.raw_data().data())[0] : t.float_data(0));
          break;
          }

        case onnx::TensorProto::UINT8:
          {
          const size_t s = raw ? rs : t.int32_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((uint8_t*) t.raw_data().data())[0] : t.int32_data(0));
          break;
          }

        case onnx::TensorProto::INT8:
          {
          const size_t s = raw ? rs : t.int32_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((int8_t*) t.raw_data().data())[0] : t.int32_data(0));
          break;
          }

        case onnx::TensorProto::UINT16:
          {
          const size_t s = raw ? (rs / 2) : t.int32_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((uint16_t*) t.raw_data().data())[0] :
              t.int32_data(0));
          break;
          }

        case onnx::TensorProto::INT16:
          {
          const size_t s = raw ? (rs / 2) : t.int32_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((int16_t*) t.raw_data().data())[0] : t.int32_data(0));
          break;
          }

        case onnx::TensorProto::INT32:
          {
          const size_t s = raw ? (rs / 4) : t.int32_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((int32_t*) t.raw_data().data())[0] : t.int32_data(0));
          break;
          }

        case onnx::TensorProto::INT64:
          {
          const size_t s = raw ? (rs / 8) : t.int64_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((int64_t*) t.raw_data().data())[0] : t.int64_data(0));
          break;
          }

        case onnx::TensorProto::DOUBLE:
          {
          const size_t s = raw ? (rs / 8) : t.double_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((double*) t.raw_data().data())[0] : t.double_data(0));
          break;
          }

        case onnx::TensorProto::UINT32:
          {
          const size_t s = raw ? (rs / 4) : t.uint64_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((uint32_t*) t.raw_data().data())[0] :
              t.uint64_data(0));
          break;
          }

        case onnx::TensorProto::UINT64:
          {
          const size_t s = raw ? (rs / 8) : t.uint64_data_size();
          if (s != 1)
            return false;

          val = eT(raw ? ((uint64_t*) t.raw_data().data())[0] :
              t.uint64_data(0));
          break;
          }

        default:
          // Unknown or unhandled type.
          return false;
      }

      return true;
    }
  }

  return false;
}

// Returns true if *only* one of tensor1Name or tensor2Name are scalars.
template<typename eT>
inline bool ExtractEitherScalar(const onnx::GraphProto& graph,
                                const std::string& tensor1Name,
                                const std::string& tensor2Name,
                                eT value)
{
  const bool e1 = ExtractScalar(graph, tensor1Name, value);
  const bool e2 = ExtractScalar(graph, tensor2Name, value);
  if (e1 && e2)
    return false;
  else if (!e1 && !e2)
    return false;

  return true;
}

} // namespace onnx_mlpack

#endif
