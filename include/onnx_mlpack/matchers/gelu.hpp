/**
 * @file gelu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the approximate GELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_HPP
#define ONNX_MLPACK_MATCHERS_GELU_HPP

namespace onnx_mlpack {

class GELUSubgraph : public Subgraph
{
 public:
  GELUSubgraph() : Subgraph({ "Gelu" }) { }

  inline const char* Name() const { return "GELU"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
