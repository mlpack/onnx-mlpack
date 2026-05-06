/**
 * @file gelu_exact.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the exact GELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_EXACT_HPP
#define ONNX_MLPACK_MATCHERS_GELU_EXACT_HPP

namespace onnx_mlpack {

class GELUExactSubgraph : public Subgraph
{
 public:
  GELUExactSubgraph() : Subgraph({ "Gelu" }) { }

  inline const char* Name() const { return "GELUExact"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
