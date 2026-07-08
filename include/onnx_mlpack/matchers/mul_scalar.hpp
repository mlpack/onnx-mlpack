/**
 * @file mul.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match a plain Mul node that multiplies a
 * scalar.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_MUL_SCALAR_HPP
#define ONNX_MLPACK_MATCHERS_MUL_SCALAR_HPP

namespace onnx_mlpack {

class MulScalarSubgraph : public Subgraph
{
 public:
  MulScalarSubgraph() : Subgraph({ "Mul" }) { }

  inline const char* Name() const { return "MulScalar"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
