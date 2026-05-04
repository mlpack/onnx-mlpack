/**
 * @file tanh.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Tanh layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_TANH_HPP
#define ONNX_MLPACK_MATCHERS_TANH_HPP

namespace onnx_mlpack {

class TanhSubgraph : public Subgraph
{
 public:
  TanhSubgraph() : Subgraph({ "Tanh" }) { }

  inline const char* Name() { return "Tanh"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
