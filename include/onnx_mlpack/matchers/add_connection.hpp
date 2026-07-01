/**
 * @file add_connection.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match an Add ONNX operation that represents
 * when the outputs of two other layers need to be added together as inputs to
 * another layer.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_ADD_CONNECTION_HPP
#define ONNX_MLPACK_MATCHERS_ADD_CONNECTION_HPP

namespace onnx_mlpack {

class AddConnectionSubgraph : public Subgraph
{
 public:
  AddConnectionSubgraph() : Subgraph({ "Add" }) { }

  inline const char* Name() const { return "AddConnection"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
