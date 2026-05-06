/**
 * @file mish.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mish layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MISH_HPP
#define ONNX_MLPACK_MATCHERS_MISH_HPP

namespace onnx_mlpack {

class MishSubgraph : public Subgraph
{
 public:
  MishSubgraph() : Subgraph({ "Mish" }) { }

  inline const char* Name() const { return "Mish"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
