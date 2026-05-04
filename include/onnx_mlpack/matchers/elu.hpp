/**
 * @file elu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the ELU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_ELU_HPP
#define ONNX_MLPACK_MATCHERS_ELU_HPP

namespace onnx_mlpack {

class ELUSubgraph : public Subgraph
{
 public:
  ELUSubgraph() : Subgraph({ "Elu" }) { }

  inline const char* Name() { return "ELU"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
