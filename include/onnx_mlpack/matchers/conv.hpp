/**
 * @file conv.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_HPP
#define ONNX_MLPACK_MATCHERS_CONV_HPP

namespace onnx_mlpack {

class ConvSubgraph : public Subgraph
{
 public:
  ConvSubgraph() : Subgraph({ "Conv" }) { }

  inline const char* Name() const { return "Conv"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;

  inline void TransferWeights(const arma::uvec& indices,
                              const onnx::GraphProto& graph,
                              std::vector<mlpack::Layer<>*>& layers)
      const override;
};

} // namespace onnx_mlpack

#endif
