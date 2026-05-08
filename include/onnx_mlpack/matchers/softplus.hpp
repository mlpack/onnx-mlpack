/**
 * @file softplus.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Softplus layer.
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
