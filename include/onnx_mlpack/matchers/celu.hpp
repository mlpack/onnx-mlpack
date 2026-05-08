/**
 * @file celu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the CELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_CELU_HPP
#define ONNX_MLPACK_MATCHERS_CELU_HPP

namespace onnx_mlpack {

class CELUSubgraph : public Subgraph
{
 public:
  CELUSubgraph() : Subgraph({ "Celu" }) { }

  inline const char* Name() const { return "CELU"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
