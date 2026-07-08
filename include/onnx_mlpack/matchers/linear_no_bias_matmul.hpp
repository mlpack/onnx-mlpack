/**
 * @file linear_no_bias_matmul.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_HPP

namespace onnx_mlpack {

class LinearNoBiasMatMulSubgraph : public Subgraph
{
 public:
  LinearNoBiasMatMulSubgraph() : Subgraph({ "MatMul" }) { }

  inline const char* Name() const { return "LinearNoBiasMatMul"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;

  inline void TransferWeights(const arma::uvec& indices,
                              const onnx::GraphProto& graph,
                              std::vector<mlpack::Layer<>*>& layer)
      const override;
};

} // namespace onnx_mlpack

#endif
