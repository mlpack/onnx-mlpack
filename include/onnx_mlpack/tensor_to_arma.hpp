/**
 * @file tensor_to_arma.hpp
 * @author Ryan Curtin
 *
 * Given a TensorProto, return an Armadillo matrix with the converted data.
 */
#ifndef ONNX_MLPACK_TENSOR_TO_ARMA_HPP
#define ONNX_MLPACK_TENSOR_TO_ARMA_HPP

#include <armadillo>
#include <onnx/onnx_pb.h>

namespace onnx_mlpack {

template<typename eT = double>
arma::Mat<eT> TensorToArma(const onnx::TensorProto& tensor)
{
  if (tensor.dims_size() > 2)
  {
    std::ostringstream oss;
    if (tensor.has_name())
      oss << "TensorToArma(): cannot convert tensor '" << tensor.name() << "' ";
    else
      oss << "TensorToArma(): cannot convert unnamed tensor ";
    oss << "with " << tensor.dims_size() << " dimensions to Armadillo; ";
    oss << "maximum number of dimensions is 2!";
    throw std::runtime_error(oss.str());
  }

  if (tensor.dims_size() == 0)
  {
    // Tensor is empty...
    return arma::Mat<eT>();
  }

  // Multi-segment tensors are currently not supported...
  if (tensor.has_segment())
  {
    std::ostringstream oss;
    if (tensor.has_name())
      oss << "TensorToArma(): cannot convert tensor '" << tensor.name() << "'";
    else
      oss << "TensorToArma(): cannot convert unnamed tensor ";
    oss << "to Armadillo; chunked tensor support is currently unimplemented.";
    throw std::runtime_error(oss.str());
  }

  if (tensor.data_location() == onnx::TensorProto::EXTERNAL)
  {
    std::ostringstream oss;
    if (tensor.has_name())
      oss << "TensorToArma(): cannot convert tensor '" << tensor.name() << "'";
    else
      oss << "TensorToArma(): cannot convert unnamed tensor ";
    oss << "to Armadillo; external data support is currently unimplemented; ";
    oss << "use an ONNX graph with weights stored in the ONNX file.";
    throw std::runtime_error(oss.str());
  }

  // ONNX stores as row-major, so we need to do an implicit transpose to get
  // column-major (which is what we need).
  const size_t c = tensor.dims(0);
  const size_t r = (tensor.dims_size() == 1) ? 1 : tensor.dims(1);

  switch (tensor.data_type())
  {
    case onnx::TensorProto::FLOAT:
      {
        float* dataPtr = (float*) (tensor.has_raw_data() ?
            (const float*) tensor.raw_data().data() :
            tensor.float_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(
            arma::fmat(dataPtr, r, c, false));
      }

    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
      {
        // According to onnx.proto, all data types with a bit-width of 8 or
        // greater are stored in a single int32_data element (so, lots of
        // zeros...).
        uint32_t* dataPtr = (uint32_t*) (tensor.has_raw_data() ?
            (const uint32_t*) tensor.raw_data().data() :
            (const uint32_t*) tensor.int32_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(
            arma::Mat<uint32_t>(dataPtr, r, c, false));
      }

    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
      {
        int32_t* dataPtr = (int32_t*) (tensor.has_raw_data() ?
            (const int32_t*) tensor.raw_data().data() :
            tensor.int32_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(
            arma::Mat<int32_t>(dataPtr, r, c, false));
      }

    case onnx::TensorProto::INT64:
      {
        int64_t* dataPtr = (int64_t*) (tensor.has_raw_data() ?
            (const int64_t*) tensor.raw_data().data() :
            tensor.int64_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(
            arma::Mat<int64_t>(dataPtr, r, c, false));
      }

    case onnx::TensorProto::DOUBLE:
      {
        double* dataPtr = (double*) (tensor.has_raw_data() ?
            (const double*) tensor.raw_data().data() :
            tensor.double_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(arma::Mat<double>(
            dataPtr, r, c, false));
      }

    case onnx::TensorProto::BOOL:
      {
        return arma::conv_to<arma::Mat<eT>>::from(arma::Mat<unsigned char>(
            (unsigned char*) tensor.raw_data().data(), r, c, false));
      }

    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      {
        uint64_t* dataPtr = (uint64_t*) (tensor.has_raw_data() ?
            (const uint64_t*) tensor.raw_data().data() :
            tensor.uint64_data().data());
        return arma::conv_to<arma::Mat<eT>>::from(arma::Mat<uint64_t>(
            dataPtr, r, c, false));
      }

    default:
      {
        std::string type = "unknown";
        switch (tensor.data_type())
        {
          case onnx::TensorProto::FLOAT16:
            type = "float16";
            break;
          case onnx::TensorProto::STRING:
            type = "string";
            break;
          case onnx::TensorProto::COMPLEX64:
            type = "complex64";
            break;
          case onnx::TensorProto::COMPLEX128:
            type = "complex128";
            break;
          case onnx::TensorProto::BFLOAT16:
            type = "bfloat16";
            break;
          case onnx::TensorProto::FLOAT8E4M3FN:
            type = "float8e4m3fn";
            break;
          case onnx::TensorProto::FLOAT8E4M3FNUZ:
            type = "float8e4m3fnuz";
            break;
          case onnx::TensorProto::FLOAT8E5M2:
            type = "float8e5m2";
            break;
          case onnx::TensorProto::FLOAT8E5M2FNUZ:
            type = "float8e5m2fnuz";
            break;
          // TODO: ifdef gating for these types, which are not available in
          // every ONNX version.
          //case onnx::TensorProto::UINT4:
          //  type = "uint4";
          //  break;
          //case onnx::TensorProto::INT4:
          //  type = "int4";
          //  break;
          //case onnx::TensorProto::FLOAT4E2M1:
          //  type = "float4e2m1";
          //  break;
          //case onnx::TensorProto::FLOAT8E8M0:
          //  type = "float8e8m0";
          //  break;
          //case onnx::TensorProto::UINT2:
          //  type = "uint2";
          //  break;
          //case onnx::TensorProto::INT2:
          //  type = "int2";
          //  break;
        }

        std::ostringstream oss;
        if (tensor.has_name())
        {
          oss << "TensorToArma(): cannot convert tensor '" << tensor.name()
              << "' ";
        }
        else
        {
          oss << "TensorToArma(): cannot convert unnamed tensor ";
        }
        oss << "to Armadillo; source data type (" << type << ") not supported!";
        throw std::runtime_error(oss.str());
      }
  }
}

} // namespace onnx_mlpack

#endif
