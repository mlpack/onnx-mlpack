/**
 * @file softplus.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Softplus layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_SOFTPLUS_HPP
#define ONNX_MLPACK_MATCHERS_SOFTPLUS_HPP

namespace onnx_mlpack {

class SoftplusSubgraph : public Subgraph
{
 public:
  SoftplusSubgraph() : Subgraph({ "Softplus" }) { }

  inline const char* Name() const { return "Softplus"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
