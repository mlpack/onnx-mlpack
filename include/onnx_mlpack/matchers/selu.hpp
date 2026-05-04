/**
 * @file selu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the SELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SELU_HPP
#define ONNX_MLPACK_MATCHERS_SELU_HPP

namespace onnx_mlpack {

class SELUSubgraph : public Subgraph
{
 public:
  SELUSubgraph() : Subgraph({ "Selu" }) { }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
