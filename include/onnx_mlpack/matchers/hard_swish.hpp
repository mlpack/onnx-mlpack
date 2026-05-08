/**
 * @file hard_swish.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSwish layer using a
 * HardSigmoid ONNX node.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SWISH_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SWISH_HPP

namespace onnx_mlpack {

class HardSwishSubgraph : public Subgraph
{
 public:
  HardSwishSubgraph() : Subgraph({ "HardSwish" }) { }

  inline const char* Name() const { return "HardSwish"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
