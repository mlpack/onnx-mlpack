/**
 * @file conv_add.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer when it is followed by
 * an Add (per-feature-map bias).
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_ADD_HPP
#define ONNX_MLPACK_MATCHERS_CONV_ADD_HPP

namespace onnx_mlpack {

class ConvAddSubgraph : public Subgraph
{
 public:
  ConvAddSubgraph() : Subgraph({ "Conv", "Add" }, { { 0, 1 } }) { }

  inline const char* Name() const { return "ConvAdd"; }

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
