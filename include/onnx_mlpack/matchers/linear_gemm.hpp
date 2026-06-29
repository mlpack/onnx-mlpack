/**
 * @file linear_gemm.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the Linear layer with a Gemm ONNX
 * operation..
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_GEMM_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_GEMM_HPP

namespace onnx_mlpack {

class LinearGemmSubgraph : public Subgraph
{
 public:
  LinearGemmSubgraph() : Subgraph({ "Gemm" }) { }

  inline const char* Name() const { return "LinearGEMM"; }

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
