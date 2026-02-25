/**
 * @file mul.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX Mul operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_MUL_HPP
#define ONNX_MLPACK_OPERATORS_MUL_HPP

#include "../helper.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddMul(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

class ScaleLayer : public mlpack::Identity<arma::mat>
{
 public:
  float scalar;
  inline ScaleLayer(float scalar) : scalar(scalar) {}

  inline void Forward(const arma::mat &input, arma::mat &output)
  {
    output = input * scalar;
  }
};

inline float FindScallingFactor(onnx::GraphProto graph, onnx::NodeProto node);

} // namespace onnx_mlpack

#include "mul_impl.hpp"

#endif
